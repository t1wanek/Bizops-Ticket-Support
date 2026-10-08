
import os, re, json
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="OpsPilot — AI Ticket Triage",
    page_icon="⚡",
    layout="wide",
)

DATA_PATH = Path(__file__).parent / "bizops_challenge_support_tickets.csv"
df = pd.read_csv(DATA_PATH)
df["created_at"] = pd.to_datetime(df["created_at"])

# ---------- AI-style operational extraction ----------
URGENT_TERMS = {
    "payroll": 30, "outage": 30, "all cards": 28, "several": 12,
    "stopped working": 28, "not working": 22, "rejected": 20,
    "failing": 20, "missing": 18, "not received": 22,
    "affecting payroll": 35, "today": 18, "tomorrow": 18,
    "this afternoon": 20, "supplier": 15, "services will": 18,
    "services are": 15, "restricted": 25, "unrecognized": 35,
    "nobody here recognizes": 40, "still cannot": 15,
}
PRODUCT_BASE = {"Payments": 8, "Cards": 7, "FX": 5, "Account": 3}

def money_exposure(text):
    s = str(text).lower().replace(",", "")
    vals = []
    # Handles $118k / $31,500 USD / $42,000 USD
    for m in re.finditer(r"\$(\d+(?:\.\d+)?)\s*([km])?", s):
        val = float(m.group(1))
        if m.group(2) == "k":
            val *= 1000
        elif m.group(2) == "m":
            val *= 1000000
        vals.append(val)
    return max(vals) if vals else 0

def extract_signals(row):
    text = f"{row.ticket_subject} {row.ticket_body}".lower()
    score = PRODUCT_BASE.get(row.product, 0)
    reasons = []
    tags = []

    for term, pts in URGENT_TERMS.items():
        if term in text:
            score += pts
            reasons.append(term)

    amount = money_exposure(text)
    if amount >= 100000:
        score += 45; tags.append(f"${amount:,.0f} exposure")
    elif amount >= 30000:
        score += 30; tags.append(f"${amount:,.0f} exposure")
    elif amount >= 10000:
        score += 18; tags.append(f"${amount:,.0f} exposure")
    elif amount > 0:
        score += 8; tags.append(f"${amount:,.0f} exposure")

    if row.monthly_revenue_usd >= 10000:
        score += 12; tags.append("high-value account")
    elif row.monthly_revenue_usd >= 5000:
        score += 7; tags.append("mid-value account")

    if row.customer_segment == "Mid-Market":
        score += 4

    if row.status == "open":
        score += 12
    elif row.status == "pending":
        score += 8

    if row.product == "Payments" and any(x in text for x in ["not received", "missing", "wire"]):
        tags.append("funds movement")
    if row.product == "Cards" and any(x in text for x in ["all cards", "several", "outage"]):
        tags.append("possible incident")
    if "payroll" in text:
        tags.append("payroll dependency")
    if "supplier" in text:
        tags.append("supplier deadline")
    if "unrecognized" in text or "nobody here recognizes" in text:
        tags.append("possible account-security issue")

    if score >= 85:
        level = "Critical"
    elif score >= 60:
        level = "High"
    elif score >= 35:
        level = "Medium"
    else:
        level = "Low"

    if "unrecognized" in text or "nobody here recognizes" in text:
        action = "Escalate to account-security / fraud queue; verify admin activity and freeze risky changes if policy allows."
    elif "outage" in text or "all cards" in text or "several online card payments" in text:
        action = "Check for active card-processing incident; link related tickets and move to incident response."
    elif "payroll" in text and ("wire" in text or "missing" in text or "not received" in text):
        action = "Escalate to payments operations immediately; trace the transfer and proactively update the customer."
    elif "supplier" in text or "today" in text or "tomorrow" in text:
        action = "Prioritize same-day operations review and confirm the next customer-visible milestone."
    elif row.product == "FX":
        action = "Route to FX/pricing support; verify rate source, timing and whether a pricing change occurred."
    else:
        action = "Route to the product queue with a clear owner, next action and customer update."

    if level == "Critical":
        response = "I’m sorry this is affecting your operations. I’ve escalated this to our specialist operations team and flagged the business impact. We’ll confirm the next step as soon as we have an update."
    elif level == "High":
        response = "Thanks for flagging this. I’ve prioritized the case based on the operational impact and am routing it to the appropriate specialist team. I’ll keep you updated on the next step."
    else:
        response = "Thanks for reaching out. I’ve reviewed the request and routed it to the appropriate team. We’ll follow up with the next step once it has been reviewed."

    return pd.Series({
        "risk_score": min(int(score), 100),
        "risk_level": level,
        "signals": ", ".join(tags[:5]) if tags else "standard request",
        "reasons": ", ".join(reasons[:5]),
        "recommended_action": action,
        "draft_response": response,
        "amount_exposure": amount,
    })

signals = df.apply(extract_signals, axis=1)
df = pd.concat([df, signals], axis=1)

# Simple related-ticket detection: same customer/product + strong body similarity.
def related_ids(row, data):
    out = []
    for _, other in data.iterrows():
        if other.ticket_id == row.ticket_id:
            continue
        if other.customer_name != row.customer_name or other.product != row.product:
            continue
        a = f"{row.ticket_subject} {row.ticket_body}".lower()
        b = f"{other.ticket_subject} {other.ticket_body}".lower()
        if SequenceMatcher(None, a, b).ratio() >= 0.52:
            out.append(other.ticket_id)
    return out

df["related_tickets"] = df.apply(lambda r: related_ids(r, df), axis=1)

# ---------- Optional real LLM ----------
def llm_triage(row):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        prompt = f"""
You are an operations triage copilot at a fintech.
Analyze this support ticket. Return strict JSON with:
risk_level (Critical/High/Medium/Low), risk_score (0-100),
signals (array of short strings), recommended_action (one sentence),
draft_response (customer-safe response, no invented facts).

Ticket:
{json.dumps({
"customer": row.customer_name,
"segment": row.customer_segment,
"monthly_revenue_usd": row.monthly_revenue_usd,
"product": row.product,
"subject": row.ticket_subject,
"body": row.ticket_body,
"status": row.status
})}
"""
        resp = client.responses.create(model=model, input=prompt)
        text = resp.output_text
        return json.loads(text)
    except Exception:
        return None

# ---------- UI ----------
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.metric-card {padding: 14px 16px; border: 1px solid #e5e7eb; border-radius: 12px; background: #fff;}
.small {color:#6b7280;font-size:0.86rem;}
.badge {padding:3px 8px;border-radius:999px;font-size:0.78rem;font-weight:600;}
</style>
""", unsafe_allow_html=True)

st.title("⚡ OpsPilot")
st.caption("AI-assisted support triage for a fintech operations team")

unresolved = df[df.status != "resolved"].copy()
critical = unresolved[unresolved.risk_level == "Critical"]
high = unresolved[unresolved.risk_level == "High"]
exposure = unresolved.monthly_revenue_usd.sum()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Unresolved tickets", len(unresolved))
c2.metric("Critical / high risk", len(pd.concat([critical, high])))
c3.metric("Monthly revenue represented", f"${exposure:,.0f}")
c4.metric("1★ historical CSAT", int((df.csat_score == 1).sum()))

st.divider()

tab1, tab2, tab3 = st.tabs(["🚦 Priority queue", "🔎 Ticket copilot", "📊 Operational insights"])

with tab1:
    st.subheader("What should Ops work on next?")
    st.caption("Priority combines business impact, operational urgency, status, product risk and explicit customer deadlines.")

    col1,col2,col3 = st.columns([1,1,2])
    with col1:
        risk_filter = st.multiselect("Risk", ["Critical","High","Medium","Low"], default=["Critical","High","Medium"])
    with col2:
        status_filter = st.multiselect("Status", sorted(df.status.unique()), default=["open","pending"])
    with col3:
        product_filter = st.multiselect("Product", sorted(df.product.unique()), default=sorted(df.product.unique()))

    q = df[
        df.risk_level.isin(risk_filter) &
        df.status.isin(status_filter) &
        df.product.isin(product_filter)
    ].copy().sort_values(["risk_score","monthly_revenue_usd"], ascending=False)

    display = q[["ticket_id","customer_name","product","status","risk_level","risk_score","signals","monthly_revenue_usd"]].copy()
    display["monthly_revenue_usd"] = display["monthly_revenue_usd"].map(lambda x: f"${x:,.0f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("**Top recommendation:** Work the highest-risk ticket first, then batch related tickets so one operational investigation resolves multiple customer contacts.")

with tab2:
    st.subheader("Ticket copilot")
    ticket_id = st.selectbox("Select a ticket", df.sort_values(["status","risk_score"], ascending=[True,False]).ticket_id.tolist())
    row = df[df.ticket_id == ticket_id].iloc[0]

    a,b,c = st.columns([1,1,1])
    a.metric("Risk", f"{row.risk_level} · {row.risk_score}/100")
    b.metric("Status", row.status.title())
    c.metric("Customer value", f"${row.monthly_revenue_usd:,.0f}/mo")

    st.markdown(f"### {row.ticket_subject}")
    st.write(row.ticket_body)
    st.caption(f"{row.customer_name} · {row.customer_segment} · {row.product} · {row.country}")

    if st.button("✨ Run AI copilot", type="primary"):
        result = llm_triage(row)
        if result:
            st.session_state["ai_result"] = result
        else:
            st.session_state["ai_result"] = {
                "risk_level": row.risk_level,
                "risk_score": row.risk_score,
                "signals": [x.strip() for x in row.signals.split(",")],
                "recommended_action": row.recommended_action,
                "draft_response": row.draft_response,
            }

    result = st.session_state.get("ai_result")
    if result:
        left,right = st.columns(2)
        with left:
            st.markdown("#### Extracted signals")
            for s in result["signals"]:
                st.write("•", s)
            st.markdown("#### Recommended next action")
            st.info(result["recommended_action"])
        with right:
            st.markdown("#### Customer-safe draft")
            st.text_area("Draft", result["draft_response"], height=170, label_visibility="collapsed")

    if row.related_tickets:
        st.markdown("#### Possible related tickets")
        st.write(", ".join(row.related_tickets))

with tab3:
    st.subheader("What the dataset is telling us")
    p = df.groupby("product").agg(
        tickets=("ticket_id","size"),
        avg_resolution_hours=("resolution_time_hours","mean"),
        avg_csat=("csat_score","mean"),
        unresolved=("status", lambda s: (s!="resolved").sum())
    ).reset_index()
    p["avg_resolution_hours"] = p["avg_resolution_hours"].round(1)
    p["avg_csat"] = p["avg_csat"].round(2)
    st.dataframe(p, use_container_width=True, hide_index=True)

    st.markdown("#### Key observations")
    obs = [
        f"**{len(unresolved)} of {len(df)} tickets are unresolved**, so queue prioritization is a live operational problem in this sample.",
        f"**Payments and Cards dominate the unresolved queue** ({(unresolved.product.isin(['Payments','Cards'])).sum()} of {len(unresolved)} tickets).",
        f"**${unresolved.monthly_revenue_usd.sum():,.0f} of monthly customer revenue is represented in unresolved cases** — useful as a prioritization signal, not a claim of revenue at risk.",
        f"**FX has the weakest historical CSAT ({df[df.product=='FX'].csat_score.mean():.2f}/5)** and the slowest average resolution time ({df[df.product=='FX'].resolution_time_hours.mean():.1f}h).",
        "Several unresolved cases contain explicit operational deadlines (payroll, supplier payment, service interruption) that a basic FIFO queue would not capture.",
    ]
    for x in obs:
        st.markdown("• " + x)

st.divider()
st.caption("Prototype note: without OPENAI_API_KEY the app uses a transparent deterministic fallback so the demo works immediately. With a key, the copilot can use an LLM for ticket-specific extraction and drafting.")
