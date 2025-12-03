# Testing Guide: Trade Decision Engine

## Where to Test

You can test the Trade Decision Engine in **two locations**:

---

## ✅ Option 1: Phase 1 - After QUANTUMTRADER (Recommended for Quick Testing)

**Location**: `app.py` around **line 1176** (right after QUANTUMTRADER results)

### Steps to Add:

1. **Open** `app.py`
2. **Find** the section after QUANTUMTRADER prompt (around line 1176)
3. **Add** the following code block:

```python
    # Trade Decision Engine Test
    st.markdown("---")
    st.markdown("##### 🎯 TRADE DECISION ENGINE")
    
    if quantum_result:
        if st.button("🔍 Evaluate Trade Decision", key="evaluate_trade_decision"):
            with st.spinner("Evaluating all trade conditions..."):
                try:
                    from tradingagents.agents.utils.trading_engine import TradeDecisionEngine
                    
                    decision_engine = TradeDecisionEngine()
                    decision = decision_engine.evaluate_trade_decision(lab_symbol, quantum_timestamp)
                    st.session_state.trade_decision = decision
                    decision_engine.close()
                    
                    st.success("✅ Trade decision evaluation complete!")
                except Exception as exc:
                    st.error(f"Trade decision evaluation failed: {exc}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Display decision results
        trade_decision = st.session_state.get("trade_decision")
        if trade_decision:
            st.markdown("###### Trade Decision Results")
            
            # Decision summary
            decision_color = "🟢" if trade_decision["trade_decision"] == "TRADE YES" else "🔴"
            st.markdown(f"### {decision_color} Decision: **{trade_decision['trade_decision']}**")
            
            if trade_decision["direction"]:
                st.info(f"**Direction**: {trade_decision['direction']}")
            
            st.markdown(f"**Reason**: {trade_decision['reason']}")
            
            # Conditions breakdown
            st.markdown("#### Condition Checks:")
            conditions = trade_decision["conditions"]
            
            # Condition 1: Composite Score
            cond1 = conditions.get("composite_score", {})
            st.markdown(f"""
            **1. Composite Score ≥ 6.5**
            - Current: {cond1.get('value', 'N/A'):.2f if cond1.get('value') else 'N/A'}
            - Threshold: {cond1.get('threshold', 6.5)}
            - Status: {'✅ PASS' if cond1.get('pass') else '❌ FAIL'}
            """)
            
            # Condition 2: Phase 1 Gates
            cond2 = conditions.get("phase1_gates", {})
            st.markdown(f"""
            **2. Phase 1 Gates**
            - Status: {'✅ PASS' if cond2.get('pass') else '❌ FAIL'}
            - Reason: {cond2.get('reason', 'N/A')}
            """)
            
            # Condition 3: R:R Ratio
            cond3 = conditions.get("rr_ratio", {})
            st.markdown(f"""
            **3. R:R Ratio ≥ 1:2**
            - Ratio: {cond3.get('ratio', 'N/A')}:1
            - Achievable: {'✅ YES' if cond3.get('achievable') else '❌ NO'}
            - Stop Loss: ${cond3.get('stop_loss', 'N/A')}
            - Target 1: ${cond3.get('target1', 'N/A')}
            - Target 2: ${cond3.get('target2', 'N/A')}
            """)
            
            # Condition 4: Position Size
            cond4 = conditions.get("position_size", {})
            st.markdown(f"""
            **4. Position Size ≤ $2,000**
            - Exposure: ${cond4.get('exposure', 'N/A')}
            - Shares: {cond4.get('recommended_shares', 'N/A')}
            - Status: {'✅ PASS' if cond4.get('calculable') else '❌ FAIL'}
            """)
            
            # Condition 5: Trend Alignment
            cond5 = conditions.get("trend_alignment", {})
            st.markdown(f"""
            **5. No Trend Conflict**
            - 5-min Trend: {cond5.get('m5_trend', 'N/A')}
            - Daily Trend: {cond5.get('daily_trend', 'N/A')}
            - Conflict: {'❌ YES' if cond5.get('conflict') else '✅ NO'}
            """)
            
            # Risk Metrics
            risk_metrics = trade_decision.get("risk_metrics", {})
            st.markdown("#### Risk Management:")
            st.markdown(f"""
            - Active Trades: {risk_metrics.get('active_trades', 0)} / {risk_metrics.get('max_concurrent', 3)}
            - Max Daily Loss: ${risk_metrics.get('max_daily_loss', 400)}
            - Can Trade: {'✅ YES' if risk_metrics.get('can_trade') else '❌ NO'}
            """)
            
            # Recommendation (if TRADE YES)
            if trade_decision["trade_decision"] == "TRADE YES":
                rec = trade_decision.get("recommendation", {})
                st.success("""
                ### ✅ TRADE YES - Ready to Execute!
                
                **Recommendation Details:**
                """)
                st.json(rec)
```

### Benefits:
- ✅ Quick test right after QUANTUMTRADER
- ✅ Uses same symbol and timestamp
- ✅ See all conditions evaluated
- ✅ Clear pass/fail indicators

---

## ✅ Option 2: Phase 3 - Trading Workflow Tab (Integrated Testing)

**Location**: `app.py` around **line 3166** (in Phase 3 Trading Workflow tab, after Step 6)

### Steps to Add:

1. **Open** `app.py`
2. **Find** the Phase 3 Trading Workflow tab section (around line 3166, after "Step 6: Final Decision")
3. **Add** as "Step 7: Trade Decision Engine":

```python
            # Step 7: Trade Decision Engine
            st.markdown("---")
            st.markdown("#### Step 7: Trade Decision Engine")
            
            st.info("""
            **Evaluate all conditions for TRADE YES decision:**
            - Composite Score ≥ 6.5
            - All Phase 1 gates passed
            - R:R ratio ≥ 1:2 achievable
            - Position size within $2,000 limit
            - No conflicting daily trend
            """)
            
            if st.button("🎯 Run Trade Decision Evaluation", key="run_trade_decision"):
                with st.spinner("Evaluating all trade conditions..."):
                    try:
                        if 'trade_decision_engine' not in st.session_state:
                            st.session_state.trade_decision_engine = TradeDecisionEngine()
                        
                        decision_engine = st.session_state.trade_decision_engine
                        decision = decision_engine.evaluate_trade_decision(selected_sym)
                        
                        st.session_state.trade_decision_result = decision
                        
                        # Display results
                        decision_result = decision
                        
                        # Decision header
                        if decision_result["trade_decision"] == "TRADE YES":
                            st.success(f"""
                            ## ✅ TRADE YES - {selected_sym}
                            
                            **Direction**: {decision_result['direction']}
                            """)
                        else:
                            st.error(f"""
                            ## ❌ NO TRADE - {selected_sym}
                            
                            **Reason**: {decision_result['reason']}
                            """)
                        
                        # Conditions breakdown
                        st.markdown("### Condition Checks:")
                        conditions = decision_result["conditions"]
                        
                        # Create a nice table of conditions
                        cond_data = []
                        cond_data.append({
                            "Condition": "1. Composite Score ≥ 6.5",
                            "Status": "✅ PASS" if conditions.get("composite_score", {}).get("pass") else "❌ FAIL",
                            "Value": f"{conditions.get('composite_score', {}).get('value', 'N/A')}"
                        })
                        cond_data.append({
                            "Condition": "2. Phase 1 Gates",
                            "Status": "✅ PASS" if conditions.get("phase1_gates", {}).get("pass") else "❌ FAIL",
                            "Value": conditions.get("phase1_gates", {}).get("reason", "N/A")
                        })
                        cond_data.append({
                            "Condition": "3. R:R Ratio ≥ 1:2",
                            "Status": "✅ PASS" if conditions.get("rr_ratio", {}).get("achievable") else "❌ FAIL",
                            "Value": f"{conditions.get('rr_ratio', {}).get('ratio', 'N/A')}:1"
                        })
                        cond_data.append({
                            "Condition": "4. Position Size ≤ $2,000",
                            "Status": "✅ PASS" if conditions.get("position_size", {}).get("calculable") else "❌ FAIL",
                            "Value": f"${conditions.get('position_size', {}).get('exposure', 'N/A')}"
                        })
                        cond_data.append({
                            "Condition": "5. No Trend Conflict",
                            "Status": "✅ PASS" if not conditions.get("trend_alignment", {}).get("conflict") else "❌ FAIL",
                            "Value": f"{conditions.get('trend_alignment', {}).get('m5_trend', 'N/A')} vs {conditions.get('trend_alignment', {}).get('daily_trend', 'N/A')}"
                        })
                        
                        st.dataframe(pd.DataFrame(cond_data), width='stretch', hide_index=True)
                        
                        # Show recommendation if TRADE YES
                        if decision_result["trade_decision"] == "TRADE YES":
                            rec = decision_result.get("recommendation", {})
                            
                            st.markdown("### 📋 Trade Recommendation:")
                            col_rec1, col_rec2, col_rec3 = st.columns(3)
                            
                            with col_rec1:
                                st.metric("Action", rec.get("action", "N/A"))
                                st.metric("Entry Price", f"${rec.get('entry_price', 0):.2f}")
                                st.metric("Stop Loss", f"${rec.get('stop_loss', 0):.2f}")
                            
                            with col_rec2:
                                st.metric("Target 1", f"${rec.get('target1', 0):.2f}")
                                st.metric("Target 2", f"${rec.get('target2', 0):.2f}")
                                st.metric("R:R Ratio", f"1:{rec.get('rr_ratio', 0):.2f}")
                            
                            with col_rec3:
                                st.metric("Position Size", f"{rec.get('position_size_shares', 0)} shares")
                                st.metric("Exposure", f"${rec.get('exposure', 0):.2f}")
                                st.metric("Risk Amount", f"${rec.get('risk_amount', 0):.2f}")
                            
                            # Save to workflow
                            workflow['trade_decision'] = decision_result
                            
                    except Exception as exc:
                        st.error(f"Trade decision evaluation failed: {exc}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Show cached results
            if 'trade_decision_result' in st.session_state:
                cached_decision = st.session_state.trade_decision_result
                if cached_decision.get("symbol") == selected_sym:
                    st.info("📋 Showing cached trade decision result. Click button above to re-evaluate.")
```

### Benefits:
- ✅ Integrated into existing workflow
- ✅ Fits naturally after all other analysis
- ✅ Shows comprehensive breakdown
- ✅ Saves to workflow session

---

## 🚀 Quick Start (Recommended)

**For fastest testing**, add **Option 1** to Phase 1:

1. Run the Streamlit app: `streamlit run app.py`
2. Navigate to **Phase 1: Foundation & Data**
3. Select a symbol and run **QUANTUMTRADER Prompt**
4. Click **"Evaluate Trade Decision"** button (you'll add this)
5. See all conditions evaluated!

---

## 📝 What You'll See

The test interface will show:

1. **Decision**: TRADE YES or NO TRADE
2. **Direction**: BUY, SELL, or CONFLICT
3. **5 Condition Checks**: Each with pass/fail status
4. **Risk Metrics**: Active trades, limits, etc.
5. **Recommendation** (if TRADE YES): Entry, stops, targets, position size

---

## 🔧 Testing Tips

1. **Test with different symbols** to see various outcomes
2. **Check failed conditions** to understand why trades are rejected
3. **Compare with QUANTUMTRADER scores** to see the relationship
4. **Review risk metrics** to understand position sizing

---

## 🐛 Troubleshooting

If you get errors:
- Make sure you have 1-minute and 5-minute data ingested
- Check that Phase 1 screening has been run
- Verify database connection for risk overlay checks
- Check console for detailed error messages

---

## 📍 Exact Line Numbers

- **Option 1**: Add after line **1176** in `app.py`
- **Option 2**: Add after line **3232** (after Step 6) in `app.py`

---

Would you like me to add either of these options to your `app.py` file now?
