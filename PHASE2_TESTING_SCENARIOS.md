# Phase 2 Testing Scenarios
## Master Data & AI Integration

---

## Overview
This document provides comprehensive testing scenarios for **Phase 2: Master Data & AI Integration** of the AlphaAnalyst Trading AI Agent.

### Test Environment Requirements
- ✅ Phase 1 completed (Database initialized, historical data ingested)
- ✅ GROQ_API_KEY configured in environment
- ✅ At least 3-5 research documents uploaded
- ✅ Market data available for all watchlist stocks

---

## 1. Master Data Dashboard Testing

### **Test 1.1: Dashboard Display**
**Objective:** Verify dashboard displays key metrics correctly

**Steps:**
1. Navigate to Phase 2 → Tab 1: Master Data Dashboard
2. Verify three metrics display:
   - Total Instruments (should show 19)
   - AI Analysis Status (should show "Active")
   - Documents Processed (should show count > 0 if documents uploaded)

**Expected Result:**
- All three metrics visible
- No error messages
- Correct counts displayed

**Pass Criteria:** ✅ All metrics display correctly

---

### **Test 1.2: Generate Master Data Summary**
**Objective:** Test AI analysis generation for all instruments

**Steps:**
1. Click "Generate Master Data Summary" button
2. Wait for analysis to complete (may take 2-5 minutes)
3. Verify four summary metrics display:
   - Total Instruments (19)
   - Bullish Signals (count ≥ 0)
   - Bearish Signals (count ≥ 0)
   - Avg Confidence (should be between 1-10)

**Expected Result:**
- Success message displayed
- All four summary metrics visible
- Data stored in session state
- No error messages

**Pass Criteria:** ✅ Summary generated successfully with all metrics

**Failure Scenarios:**
- ❌ API key error → Check GROQ_API_KEY
- ❌ Timeout error → Check network connection
- ❌ Empty results → Check if market data exists

---

## 2. AI Document Analysis Testing

### **Test 2.1: Upload PDF Document**
**Objective:** Upload and process a PDF research document

**Steps:**
1. Navigate to Phase 2 → Tab 2: AI Document Analysis
2. Upload a PDF file (sample: research report on NVDA)
3. Select symbol: "NVDA"
4. Enter title: "NVIDIA Q4 Earnings Analysis"
5. Select document type: "research"
6. Click "Upload & Analyze"

**Expected Result:**
- Document uploads successfully
- AI analysis completes
- Analysis results displayed with:
  - Document analysis text
  - Sentiment (Bullish/Bearish/Neutral)
  - Confidence score (1-10)
  - Bullish signals count
  - Bearish signals count

**Pass Criteria:** ✅ PDF processed and analyzed successfully

---

### **Test 2.2: Upload DOCX Document**
**Objective:** Test DOCX file processing

**Steps:**
1. Upload a DOCX file
2. Select symbol: "TSLA"
3. Enter title: "Tesla Market Analysis"
4. Select document type: "analysis"
5. Click "Upload & Analyze"

**Expected Result:**
- DOCX file processes correctly
- Text extracted successfully
- AI analysis completes

**Pass Criteria:** ✅ DOCX processed successfully

---

### **Test 2.3: Upload TXT Document**
**Objective:** Test plain text file processing

**Steps:**
1. Upload a TXT file
2. Select symbol: "AAPL"
3. Enter title: "Apple Financial Report"
4. Click "Upload & Analyze"

**Expected Result:**
- TXT file processes correctly
- AI analysis completes

**Pass Criteria:** ✅ TXT processed successfully

---

### **Test 2.4: Upload Document Without Symbol**
**Objective:** Test document upload without symbol selection

**Steps:**
1. Upload a research document
2. Leave symbol selection empty
3. Enter title
4. Click "Upload & Analyze"

**Expected Result:**
- Warning message: "Please select a symbol and enter a title"
- Upload blocked

**Pass Criteria:** ✅ Validation prevents upload

---

### **Test 2.5: Multiple Document Analysis**
**Objective:** Test analysis of multiple documents for same symbol

**Steps:**
1. Upload 3 different research documents for "NVDA"
2. Analyze each document
3. Navigate to "Recent Document Analyses" section

**Expected Result:**
- All 3 analyses visible
- Each analysis shows different results
- Expandable sections work correctly

**Pass Criteria:** ✅ Multiple analyses displayed correctly

---

## 3. Instrument Profiles Testing

### **Test 3.1: Generate Instrument Profile**
**Objective:** Test comprehensive instrument profile generation

**Steps:**
1. Navigate to Phase 2 → Tab 3: Instrument Profiles
2. Select symbol: "NVDA"
3. Click "Generate Profile"
4. Wait for analysis (1-2 minutes)

**Expected Result:**
- Profile generated successfully
- Two-column layout displayed:
  - **Left Column: Market Data**
    - Current Price (dollar amount)
    - 30-day Change (percentage)
    - Volume (number)
    - 30-day High (dollar amount)
    - 30-day Low (dollar amount)
  - **Right Column: AI Analysis**
    - Overall Assessment (Bullish/Bearish/Neutral)
    - Recommendation (BUY/SELL/HOLD)
    - Confidence (1-10)
    - Detailed Analysis text

**Pass Criteria:** ✅ Profile generated with all data visible

---

### **Test 3.2: Document Insights Integration**
**Objective:** Test display of document insights in instrument profile

**Prerequisites:**
- Upload at least 2 research documents for the symbol before running this test

**Steps:**
1. Generate profile for "NVDA"
2. Scroll to "Document Insights" section

**Expected Result:**
- Document insights section visible
- Each uploaded document shows:
  - Document filename
  - Sentiment
  - Confidence score
  - Bullish signals list
  - Bearish signals list

**Pass Criteria:** ✅ Document insights integrated and displayed

---

### **Test 3.3: Profile Generation for Multiple Symbols**
**Objective:** Test profile generation for different stocks

**Steps:**
1. Generate profile for: "AAPL"
2. Generate profile for: "TSLA"
3. Generate profile for: "MSFT"

**Expected Result:**
- Each profile generates successfully
- Data varies by symbol
- No cached/stale data

**Pass Criteria:** ✅ All profiles generate with unique data

---

## 4. Research Insights Dashboard Testing

### **Test 4.1: Portfolio Overview Display**
**Objective:** Test portfolio insights visualization

**Prerequisites:**
- Must generate master data summary first (Test 1.2)

**Steps:**
1. Run Test 1.2 (Generate Master Data Summary)
2. Navigate to Phase 2 → Tab 4: Research Insights
3. Verify portfolio overview displays

**Expected Result:**
- Dataframe showing all 19 instruments with:
  - Symbol
  - Sentiment
  - Recommendation
  - Confidence
  - Documents count
- Sentiment distribution bar chart
- Recommendation distribution bar chart

**Pass Criteria:** ✅ All visualizations display correctly

---

### **Test 4.2: Dashboard Without Summary**
**Objective:** Test behavior when no summary generated

**Steps:**
1. Navigate to Research Insights tab
2. WITHOUT running Generate Master Data Summary

**Expected Result:**
- Info message: "Generate master data summary first to see research insights"
- No data displayed

**Pass Criteria:** ✅ Helpful message displayed

---

## 5. Error Handling & Edge Cases

### **Test 5.1: Invalid API Key**
**Objective:** Test error handling for invalid GROQ API key

**Steps:**
1. Set invalid GROQ_API_KEY in environment
2. Attempt to generate master data summary
3. Observe error handling

**Expected Result:**
- Error message displayed
- Application doesn't crash
- User-friendly error explanation

**Pass Criteria:** ✅ Graceful error handling

---

### **Test 5.2: Empty Document Analysis**
**Objective:** Test handling of empty document content

**Steps:**
1. Upload a document with minimal/no content
2. Attempt to analyze

**Expected Result:**
- Analysis completes or shows appropriate message
- No crash

**Pass Criteria:** ✅ Handles empty content gracefully

---

### **Test 5.3: Network Timeout**
**Objective:** Test handling of API timeout

**Steps:**
1. Disconnect network mid-analysis
2. Attempt to generate analysis

**Expected Result:**
- Timeout error handled
- Error message displayed
- Application remains stable

**Pass Criteria:** ✅ Timeout handled gracefully

---

## 6. Performance Testing

### **Test 6.1: Single Document Analysis Speed**
**Objective:** Measure analysis time for single document

**Steps:**
1. Upload medium-sized PDF (100-200 pages)
2. Measure time from click to completion
3. Record results

**Expected Result:**
- Analysis completes within 60 seconds
- Progress indicator visible

**Pass Criteria:** ✅ Completion time < 60 seconds

---

### **Test 6.2: Master Data Summary Performance**
**Objective:** Measure performance for generating all instrument profiles

**Steps:**
1. Click "Generate Master Data Summary"
2. Measure time from click to completion

**Expected Result:**
- Summary completes within 5 minutes
- Progress indicator visible

**Pass Criteria:** ✅ Completion time < 5 minutes

---

## 7. Integration Testing

### **Test 7.1: Data Flow from Phase 1 to Phase 2**
**Objective:** Verify Phase 1 data is accessible in Phase 2

**Steps:**
1. Verify Phase 1 data ingestion completed
2. Navigate to Phase 2
3. Generate instrument profile for any symbol

**Expected Result:**
- Market data available from database
- Historical data displays correctly
- No "data not found" errors

**Pass Criteria:** ✅ Phase 1 data integrated successfully

---

### **Test 7.2: Document Management Integration**
**Objective:** Verify uploaded documents are accessible

**Steps:**
1. Upload documents in Phase 1
2. Navigate to Phase 2 → Instrument Profiles
3. Generate profile for symbol with uploaded documents

**Expected Result:**
- Document insights appear in profile
- Content from Phase 1 documents displayed
- No document not found errors

**Pass Criteria:** ✅ Documents integrated correctly

---

## Test Summary Checklist

### Critical Tests (Must Pass for Production)
- ✅ Test 1.2: Generate Master Data Summary
- ✅ Test 2.1: Upload PDF Document
- ✅ Test 3.1: Generate Instrument Profile
- ✅ Test 4.1: Portfolio Overview Display

### High Priority Tests
- ✅ Test 2.2: Upload DOCX Document
- ✅ Test 2.4: Upload Without Symbol (Validation)
- ✅ Test 3.2: Document Insights Integration
- ✅ Test 5.1: Invalid API Key Handling

### Medium Priority Tests
- ✅ Test 2.3: Upload TXT Document
- ✅ Test 2.5: Multiple Document Analysis
- ✅ Test 3.3: Profile Generation for Multiple Symbols
- ✅ Test 6.1: Single Document Analysis Speed

### Optional Tests
- ⚪ Test 5.2: Empty Document Analysis
- ⚪ Test 5.3: Network Timeout
- ⚪ Test 6.2: Master Data Summary Performance
- ⚪ Test 4.2: Dashboard Without Summary

---

## Test Execution Template

### Session Information
- **Tester Name:** _______________
- **Date:** _______________
- **Environment:** _______________
- **Phase 2 Commit:** _______________

### Test Results
| Test ID | Test Name | Status | Notes |
|---------|-----------|--------|-------|
| 1.1 | Dashboard Display | ✅/❌ | |
| 1.2 | Generate Master Data Summary | ✅/❌ | |
| 2.1 | Upload PDF Document | ✅/❌ | |
| 2.2 | Upload DOCX Document | ✅/❌ | |
| 2.3 | Upload TXT Document | ✅/❌ | |
| 2.4 | Upload Without Symbol | ✅/❌ | |
| 2.5 | Multiple Document Analysis | ✅/❌ | |
| 3.1 | Generate Instrument Profile | ✅/❌ | |
| 3.2 | Document Insights Integration | ✅/❌ | |
| 3.3 | Profile Generation Multiple Symbols | ✅/❌ | |
| 4.1 | Portfolio Overview Display | ✅/❌ | |
| 4.2 | Dashboard Without Summary | ✅/❌ | |
| 5.1 | Invalid API Key | ✅/❌ | |
| 6.1 | Single Document Analysis Speed | ✅/❌ | |

### Overall Status
- **Critical Tests:** X/4 passed
- **Overall Pass Rate:** X/14 (XX%)
- **Ready for Production:** Yes/No

---

## Known Issues & Troubleshooting

### Issue 1: AI Analysis Timeout
**Symptom:** Analysis takes > 5 minutes or times out
**Solution:** Check GROQ_API_KEY, ensure network connectivity, reduce batch size

### Issue 2: No Data in Instrument Profile
**Symptom:** Profile shows "N/A" for all fields
**Solution:** Verify Phase 1 data ingestion completed, check database connection

### Issue 3: Document Upload Fails
**Symptom:** Document upload returns error
**Solution:** Check file size (max 10MB), verify file format (PDF/DOCX/TXT), check database permissions

### Issue 4: Missing AI Analysis Results
**Symptom:** Analysis completes but no results displayed
**Solution:** Check GROQ_API_KEY, verify API quota not exceeded, check console for errors

---

## Success Criteria Summary

### Phase 2 is Ready for Production When:
1. ✅ All critical tests pass
2. ✅ High priority tests pass
3. ✅ Master Data Summary generates successfully for all 19 instruments
4. ✅ AI analysis works for all document types (PDF, DOCX, TXT)
5. ✅ Instrument profiles display comprehensive data
6. ✅ Research Insights dashboard visualizes data correctly
7. ✅ Error handling works for all known failure scenarios
8. ✅ Performance meets requirements (< 60s single doc, < 5min summary)

### Phase 2 is Ready for Phase 3 When:
- ✅ All Phase 2 features tested and working
- ✅ Phase 1 data seamlessly integrated
- ✅ AI analysis results available for Phase 3 trading engine
- ✅ Document insights accessible for decision-making
- ✅ Complete test documentation reviewed and signed off

---

**Document Version:** 1.0  
**Last Updated:** 2024  
**Author:** AlphaAnalyst Development Team

