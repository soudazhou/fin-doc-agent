#!/usr/bin/env python3
"""
Generate a synthetic Apple Q3 2024 earnings report PDF.

This creates a realistic financial document that aligns with the golden
dataset in data/eval/golden_datasets/default.json. All figures are
synthetic but internally consistent.

Usage:
    uv run python scripts/generate_sample_pdf.py

Output:
    data/samples/apple_q3_2024_earnings.pdf
"""

from pathlib import Path

from fpdf import FPDF


class EarningsReport(FPDF):
    """Custom PDF with headers and footers for a financial report."""

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Apple Inc.  - Q3 Fiscal Year 2024 Earnings Report", 0, 1, "C")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(
            0, 10, f"Page {self.page_no()}/{{nb}} | Synthetic data for demo purposes",
            0, 0, "C",
        )

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 0, 0)
        self.ln(6)
        self.cell(0, 10, title, 0, 1)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)

    def subsection_title(self, title: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(40, 40, 40)
        self.ln(3)
        self.cell(0, 8, title, 0, 1)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def table_row(self, cells: list[str], bold: bool = False):
        style = "B" if bold else ""
        self.set_font("Helvetica", style, 9)
        col_w = 190 / len(cells)
        for cell in cells:
            self.cell(col_w, 6, cell, 1, 0, "C")
        self.ln()


def generate_report():
    pdf = EarningsReport()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # =========================================================================
    # Page 1: Cover & Financial Highlights
    # =========================================================================
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.ln(20)
    pdf.cell(0, 15, "Apple Inc.", 0, 1, "C")
    pdf.set_font("Helvetica", "", 16)
    pdf.cell(0, 10, "Quarterly Earnings Report", 0, 1, "C")
    pdf.cell(0, 10, "Q3 Fiscal Year 2024 (Quarter Ended June 29, 2024)", 0, 1, "C")
    pdf.ln(10)

    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(
        0, 8,
        "SYNTHETIC DATA  - Generated for Financial Document Q&A Agent demo",
        0, 1, "C",
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(15)

    pdf.section_title("Financial Highlights")
    pdf.body_text(
        "Apple Inc. reported total net revenue of $85.8 billion for the "
        "third fiscal quarter of 2024, compared to $81.8 billion in Q3 "
        "2023, an increase of approximately 5% year over year."
    )
    pdf.body_text(
        "Gross margin for Q3 2024 was 46.3%, compared to 44.5% in Q3 "
        "2023. The improvement was driven by favorable product mix and "
        "cost efficiencies in the supply chain."
    )
    pdf.body_text(
        "Diluted earnings per share were $1.40, an 11% increase from "
        "the $1.26 reported in the prior-year quarter, driven by higher "
        "revenue and improved operating margins."
    )
    pdf.body_text(
        "Basic earnings per share were $1.42 and diluted earnings per "
        "share were $1.40 for Q3 2024. In Q3 2023, basic EPS was $1.27 "
        "and diluted EPS was $1.26. The weighted average diluted share "
        "count decreased to 15.41 billion from 15.70 billion due to the "
        "share repurchase program."
    )

    # =========================================================================
    # Page 2: Revenue by Segment
    # =========================================================================
    pdf.add_page()
    pdf.section_title("Revenue by Product Segment")

    pdf.table_row(["Segment", "Q3 2024", "Q3 2023", "YoY Change"], bold=True)
    pdf.table_row(["iPhone", "$39.3B", "$39.6B", "-1%"])
    pdf.table_row(["Mac", "$7.0B", "$6.8B", "+2%"])
    pdf.table_row(["iPad", "$7.2B", "$5.8B", "+24%"])
    pdf.table_row(["Wearables, Home & Acc.", "$8.1B", "$8.3B", "-2%"])
    pdf.table_row(["Services", "$24.2B", "$21.2B", "+14%"])
    pdf.table_row(["Total", "$85.8B", "$81.8B", "+5%"], bold=True)
    pdf.ln(4)

    pdf.body_text(
        "iPhone revenue was $39.3 billion, compared to $39.6 billion in "
        "Q3 2023, representing a modest decline of approximately 1%. The "
        "iPhone 15 family continued to perform well despite typical "
        "seasonal patterns."
    )
    pdf.body_text(
        "Mac revenue was $7.0 billion, up 2% year over year, driven by "
        "the M3 chip family."
    )
    pdf.body_text(
        "iPad revenue was $7.2 billion, a 24% increase driven by the new "
        "M4 iPad Pro launch."
    )
    pdf.body_text(
        "Wearables, Home, and Accessories revenue was $8.1 billion, down "
        "2% from the prior year. Apple Vision Pro launched during the "
        "quarter but was offset by softness in Apple Watch."
    )
    pdf.body_text(
        "Services revenue reached an all-time record of $24.2 billion, "
        "up 14% year over year. The segment includes the App Store, "
        "Apple Music, iCloud, Apple TV+, AppleCare, and licensing revenue."
    )

    # =========================================================================
    # Page 3: Geographic Revenue & Operating Expenses
    # =========================================================================
    pdf.add_page()
    pdf.section_title("Revenue by Geographic Region")

    pdf.table_row(["Region", "Q3 2024", "Q3 2023", "YoY Change"], bold=True)
    pdf.table_row(["Americas", "$37.7B", "$36.6B", "+3%"])
    pdf.table_row(["Europe", "$21.9B", "$20.5B", "+7%"])
    pdf.table_row(["Greater China", "$14.7B", "$15.6B", "-6%"])
    pdf.table_row(["Japan", "$5.1B", "$4.8B", "+6%"])
    pdf.table_row(["Rest of Asia Pacific", "$6.4B", "$5.7B", "+12%"])
    pdf.table_row(["Total", "$85.8B", "$81.8B", "+5%"], bold=True)
    pdf.ln(4)

    pdf.body_text(
        "Americas revenue was $37.7 billion, up 3% year over year."
    )
    pdf.body_text(
        "Europe revenue was $21.9 billion, up 7% year over year."
    )
    pdf.body_text(
        "Greater China revenue was $14.7 billion, down 6% from $15.6 "
        "billion in Q3 2023. The decline was attributed to increased "
        "competition from domestic smartphone manufacturers and "
        "macroeconomic headwinds."
    )
    pdf.body_text(
        "Japan revenue was $5.1 billion, up 6% year over year."
    )
    pdf.body_text(
        "Rest of Asia Pacific revenue was $6.4 billion, up 12% year "
        "over year."
    )

    pdf.section_title("Operating Expenses")
    pdf.body_text(
        "Total operating expenses for Q3 2024 were $14.3 billion. "
        "Research and development expenses were $7.9 billion, reflecting "
        "continued investment in AI and mixed reality technologies. "
        "Selling, general, and administrative expenses were $6.4 billion."
    )
    pdf.body_text(
        "R&D expenses were $7.9 billion, up 8% from $7.3 billion in Q3 "
        "2023, reflecting investment in AI and mixed reality. SG&A "
        "expenses were $6.4 billion, up 3% from $6.2 billion in Q3 2023."
    )
    pdf.body_text(
        "R&D as a percentage of revenue increased to 9.2% from 8.9% in "
        "the prior year quarter."
    )

    # =========================================================================
    # Page 4: Income Statement & Cash Flow
    # =========================================================================
    pdf.add_page()
    pdf.section_title("Condensed Income Statement")

    pdf.table_row(["Item", "Q3 2024", "Q3 2023"], bold=True)
    pdf.table_row(["Total Revenue", "$85.8B", "$81.8B"])
    pdf.table_row(["Cost of Revenue", "$46.1B", "$45.4B"])
    pdf.table_row(["Gross Profit", "$39.7B", "$36.4B"])
    pdf.table_row(["Gross Margin", "46.3%", "44.5%"])
    pdf.table_row(["R&D Expenses", "$7.9B", "$7.3B"])
    pdf.table_row(["SG&A Expenses", "$6.4B", "$6.2B"])
    pdf.table_row(["Total Operating Expenses", "$14.3B", "$13.5B"])
    pdf.table_row(["Operating Income", "$25.4B", "$22.9B"])
    pdf.table_row(["Operating Margin", "29.6%", "28.0%"])
    pdf.table_row(["Net Income", "$21.4B", "$19.8B"])
    pdf.table_row(["Net Margin", "24.9%", "24.2%"])
    pdf.table_row(["Diluted EPS", "$1.40", "$1.26"])
    pdf.table_row(["Basic EPS", "$1.42", "$1.27"])
    pdf.ln(4)

    pdf.section_title("Cash Flow Statement")
    pdf.body_text(
        "Operating cash flow was $28.9 billion for Q3 2024. Capital "
        "expenditures were $1.8 billion, resulting in free cash flow of "
        "$27.1 billion."
    )
    pdf.body_text(
        "The company returned over $29 billion to shareholders, including "
        "$26 billion in share repurchases and $3.8 billion in dividends."
    )
    pdf.body_text(
        "Acquisitions totaled $0.4 billion during the quarter. Net "
        "decrease in cash was $2.1 billion."
    )

    pdf.table_row(["Cash Flow Item", "Q3 2024"], bold=True)
    pdf.table_row(["Operating Cash Flow", "$28.9B"])
    pdf.table_row(["Capital Expenditures", "-$1.8B"])
    pdf.table_row(["Free Cash Flow", "$27.1B"])
    pdf.table_row(["Share Repurchases", "-$26.0B"])
    pdf.table_row(["Dividends Paid", "-$3.8B"])
    pdf.table_row(["Acquisitions", "-$0.4B"])
    pdf.table_row(["Net Change in Cash", "-$2.1B"])

    # =========================================================================
    # Page 5: Balance Sheet & Margin Trends
    # =========================================================================
    pdf.add_page()
    pdf.section_title("Balance Sheet Highlights")

    pdf.body_text(
        "Cash and cash equivalents at the end of Q3 2024 totaled $29.8 "
        "billion. Including short-term and long-term marketable "
        "securities, total cash and investments were approximately "
        "$153 billion."
    )
    pdf.body_text(
        "Long-term debt at the end of Q3 2024 was $95.3 billion, a "
        "decrease from $98.1 billion at the end of fiscal year 2023. The "
        "company continues to manage its debt portfolio through a "
        "combination of fixed and floating rate instruments."
    )
    pdf.body_text(
        "Net cash position was approximately $58 billion, calculated as "
        "$153 billion in cash and securities less $95.3 billion in "
        "long-term debt."
    )

    pdf.table_row(["Balance Sheet Item", "Q3 2024"], bold=True)
    pdf.table_row(["Cash and Equivalents", "$29.8B"])
    pdf.table_row(["Total Cash & Securities", "$153.0B"])
    pdf.table_row(["Accounts Receivable", "$22.3B"])
    pdf.table_row(["Total Current Assets", "$128.5B"])
    pdf.table_row(["Total Assets", "$364.8B"])
    pdf.table_row(["Current Liabilities", "$120.1B"])
    pdf.table_row(["Long-term Debt", "$95.3B"])
    pdf.table_row(["Total Liabilities", "$279.4B"])
    pdf.table_row(["Shareholders' Equity", "$85.4B"])
    pdf.ln(4)

    pdf.subsection_title("Key Financial Ratios")
    pdf.body_text(
        "Gross margin: 46.3%. Operating margin: 29.6% ($25.4B operating "
        "income on $85.8B revenue). Net profit margin: 24.9% ($21.4B "
        "net income on $85.8B revenue)."
    )
    pdf.body_text(
        "Return on equity: approximately 25.1%. Debt-to-equity ratio: "
        "1.12 ($95.3B debt / $85.4B equity). Current ratio: 1.07 "
        "($128.5B current assets / $120.1B current liabilities)."
    )
    pdf.body_text(
        "Trailing P/E ratio: approximately 34.2 based on TTM EPS of $6.57."
    )

    pdf.subsection_title("Quarterly Gross Margin Trend")
    pdf.body_text(
        "Quarterly gross margin trend: Q4 2023 was 45.2%, Q1 2024 was "
        "45.9%, Q2 2024 was 46.6%, and Q3 2024 was 46.3%. The "
        "year-over-year improvement from 44.5% in Q3 2023 to 46.3% in "
        "Q3 2024 was driven by product mix shift toward higher-margin "
        "Services and cost efficiencies."
    )

    # =========================================================================
    # Page 6: Capital Returns, Guidance & AI Strategy
    # =========================================================================
    pdf.add_page()
    pdf.section_title("Capital Returns")
    pdf.body_text(
        "During Q3 2024, the company repurchased approximately $26 "
        "billion of its common stock. The Board of Directors authorized "
        "an additional $110 billion in share repurchases earlier in the "
        "fiscal year."
    )
    pdf.body_text(
        "The Board declared a quarterly cash dividend of $0.25 per "
        "share, payable on August 15, 2024. This represents a 4% "
        "increase over the prior year's dividend of $0.24 per share."
    )
    pdf.body_text(
        "Total capital deployment including buybacks, R&D, and dividends "
        "was approximately $37.7 billion. Buybacks represented 69% of "
        "total capital deployment, R&D received 21%, and dividends "
        "accounted for 10%."
    )

    pdf.section_title("Forward Guidance  - Q4 2024")
    pdf.body_text(
        "For Q4 2024, management expects revenue between $89 billion "
        "and $93 billion. Gross margin is expected to be between 46% "
        "and 47%. Operating expenses are expected to be between $14.2 "
        "billion and $14.4 billion."
    )

    pdf.section_title("Management Commentary  - AI Strategy")
    pdf.body_text(
        'CEO Tim Cook emphasized Apple Intelligence as "a defining '
        "moment for Apple,\" noting the company's focus on privacy-"
        "preserving on-device AI models."
    )
    pdf.body_text(
        "The partnership with OpenAI enables cloud-based generative AI "
        "features while maintaining Apple's privacy standards."
    )
    pdf.body_text(
        "Apple Intelligence is expected to launch with iOS 18, iPadOS "
        "18, and macOS Sequoia in Fall 2024."
    )
    pdf.body_text(
        "R&D spending of $7.9 billion reflects the company's significant "
        "investment in artificial intelligence and machine learning "
        "capabilities."
    )
    pdf.body_text(
        "New AI-focused developer APIs were announced at WWDC, allowing "
        "third-party apps to leverage Apple Intelligence features."
    )

    # =========================================================================
    # Page 7: Risk Factors
    # =========================================================================
    pdf.add_page()
    pdf.section_title("Risk Factors")
    pdf.body_text(
        "The company faces intense competition in all of its markets, "
        "particularly from Samsung, Google, and emerging Chinese "
        "manufacturers in smartphones."
    )
    pdf.body_text(
        "Supply chain risks remain significant due to geographic "
        "concentration of manufacturing in China and Southeast Asia."
    )
    pdf.body_text(
        "Regulatory risks include ongoing antitrust investigations by "
        "the European Commission and the US Department of Justice."
    )
    pdf.body_text(
        "Approximately 60% of revenue is generated outside the United "
        "States, exposing the company to foreign exchange fluctuations."
    )
    pdf.body_text(
        "The company depends on third-party manufacturers, principally "
        "Foxconn and TSMC, for the production of its products."
    )
    pdf.body_text(
        "Cybersecurity threats continue to evolve, requiring ongoing "
        "investment in security infrastructure."
    )

    pdf.section_title("Legal Disclaimer")
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(
        0, 5,
        "IMPORTANT: This document contains SYNTHETIC financial data "
        "generated for demonstration purposes only. It is designed to "
        "test a Financial Document Q&A system and does NOT represent "
        "actual Apple Inc. financial results. Do not use this document "
        "for investment decisions. All figures are fictional but "
        "structured to be internally consistent for evaluation testing.",
    )

    # =========================================================================
    # Save
    # =========================================================================
    output_dir = Path("data/samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "apple_q3_2024_earnings.pdf"
    pdf.output(str(output_path))
    print(f"Generated: {output_path} ({output_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    generate_report()
