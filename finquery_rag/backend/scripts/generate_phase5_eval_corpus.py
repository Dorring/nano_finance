#!/usr/bin/env python3
"""generate_phase5_eval_corpus.py

Generates a synthetic but complete financial evaluation corpus for Phase 5.

The corpus consists of 18 synthetic financial report documents (6 per
partition: dev, calibration, sealed) written as Markdown files under
``eval_corpus/phase5/{partition}/``. Each document contains four pages
with realistic financial data: company overview, income statement, balance
sheet, and cash flow statement.

A corpus manifest JSON file is written to
``eval_corpus/phase5/corpus-manifest.json`` recording each document's
filename, partition, company, period, SHA-256 digest, page count, source,
license, redistribution flag, and download timestamp.

Determinism contract
--------------------
All financial figures are hardcoded — the script does not use a random
number generator. Two runs of this script on the same source file produce
byte-identical output. Existing files are overwritten (idempotent).

Uniqueness contract
-------------------
No (metric, period, value) tuple appears in more than one partition. This
is verified at runtime before any file is written. Company names, revenue
figures, and report periods are all distinct across partitions.

Run from the ``finquery_rag/backend/`` directory::

    python scripts/generate_phase5_eval_corpus.py

Exit codes
----------
    0  - corpus generated successfully
    1  - an unrecoverable error occurred (uniqueness violation, write failure)
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Path layout
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent  # .../backend
CORPUS_DIR = ROOT_DIR / "eval_corpus" / "phase5"

# ---------------------------------------------------------------------------
# Corpus metadata constants
# ---------------------------------------------------------------------------
PARTITIONS: tuple[str, ...] = ("dev", "calibration", "sealed")
PAGE_COUNT = 4
SOURCE = "synthetic"
LICENSE = "internal"
REDISTRIBUTION_ALLOWED = False
DOWNLOADED_AT = "2026-07-23"


# ---------------------------------------------------------------------------
# Company financial data (hardcoded for determinism and uniqueness)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CompanyFinancials:
    """All financial figures for one company report, in 元 unless noted.

    Primary-period figures cover the report year; comparison-period figures
    cover the prior year. Derived metrics (gross profit, margins, debt ratio,
    revenue growth, equity) are computed from the raw figures.
    """

    letter: str
    name: str
    sector: str
    stock_code: str
    auditor: str
    period: int
    prev_period: int
    # Income statement (primary period, in 元)
    revenue: int
    cost_of_revenue: int
    operating_profit: int
    net_profit: int
    # Balance sheet (primary period, in 元)
    total_assets: int
    current_assets: int
    total_liabilities: int
    # Cash flow (primary period, in 元)
    operating_cash_flow: int
    investing_cash_flow: int
    financing_cash_flow: int
    # Comparison period (in 元)
    prev_revenue: int
    prev_net_profit: int
    prev_total_assets: int
    prev_total_liabilities: int

    @property
    def gross_profit(self) -> int:
        """Gross profit = revenue - cost of revenue."""
        return self.revenue - self.cost_of_revenue

    @property
    def gross_margin(self) -> float:
        """Gross margin as a percentage."""
        return self.gross_profit / self.revenue * 100

    @property
    def net_margin(self) -> float:
        """Net margin as a percentage."""
        return self.net_profit / self.revenue * 100

    @property
    def debt_ratio(self) -> float:
        """Debt-to-asset ratio as a percentage."""
        return self.total_liabilities / self.total_assets * 100

    @property
    def equity(self) -> int:
        """Owners' equity = total assets - total liabilities."""
        return self.total_assets - self.total_liabilities

    @property
    def revenue_growth(self) -> float:
        """Year-over-year revenue growth as a percentage."""
        return (self.revenue - self.prev_revenue) / self.prev_revenue * 100

    @property
    def filename(self) -> str:
        """Stable filename like ``company_A_annual_report_2023.md``."""
        return f"company_{self.letter}_annual_report_{self.period}.md"


# Dev partition: Companies A-F, period 2023 (comparison 2022)
COMPANY_DATA: dict[str, list[CompanyFinancials]] = {
    "dev": [
        CompanyFinancials(
            letter="A", name="东方科技", sector="信息技术", stock_code="300001",
            auditor="中诚信会计师事务所", period=2023, prev_period=2022,
            revenue=1_234_567_890, cost_of_revenue=740_740_734,
            operating_profit=185_185_184, net_profit=111_111_110,
            total_assets=2_469_135_780, current_assets=987_654_312,
            total_liabilities=1_111_111_101,
            operating_cash_flow=148_148_135,
            investing_cash_flow=-49_382_716,
            financing_cash_flow=74_074_067,
            prev_revenue=1_098_765_432, prev_net_profit=87_654_321,
            prev_total_assets=2_197_530_864, prev_total_liabilities=987_654_321,
        ),
        CompanyFinancials(
            letter="B", name="南方制造", sector="装备制造", stock_code="300002",
            auditor="大华会计师事务所", period=2023, prev_period=2022,
            revenue=2_345_678_901, cost_of_revenue=1_640_000_000,
            operating_profit=351_851_852, net_profit=234_567_890,
            total_assets=3_703_703_704, current_assets=1_481_481_481,
            total_liabilities=1_851_851_852,
            operating_cash_flow=281_481_481,
            investing_cash_flow=-93_827_156,
            financing_cash_flow=140_740_740,
            prev_revenue=2_123_456_789, prev_net_profit=198_765_432,
            prev_total_assets=3_456_789_012, prev_total_liabilities=1_728_395_061,
        ),
        CompanyFinancials(
            letter="C", name="北方能源", sector="能源电力", stock_code="300003",
            auditor="立信会计师事务所", period=2023, prev_period=2022,
            revenue=3_456_789_012, cost_of_revenue=2_560_000_000,
            operating_profit=432_098_761, net_profit=345_678_901,
            total_assets=5_555_555_555, current_assets=2_222_222_222,
            total_liabilities=3_024_691_358,
            operating_cash_flow=416_666_667,
            investing_cash_flow=-138_888_889,
            financing_cash_flow=208_333_333,
            prev_revenue=3_210_987_654, prev_net_profit=301_234_567,
            prev_total_assets=5_246_913_580, prev_total_liabilities=2_870_370_370,
        ),
        CompanyFinancials(
            letter="D", name="西方医药", sector="医药制造", stock_code="300004",
            auditor="天健会计师事务所", period=2023, prev_period=2022,
            revenue=4_567_890_123, cost_of_revenue=3_200_000_000,
            operating_profit=685_183_519, net_profit=502_467_914,
            total_assets=7_654_320_987, current_assets=3_061_728_395,
            total_liabilities=3_827_160_494,
            operating_cash_flow=570_987_654,
            investing_cash_flow=-190_329_218,
            financing_cash_flow=285_493_827,
            prev_revenue=4_321_098_765, prev_net_profit=456_790_123,
            prev_total_assets=7_283_950_617, prev_total_liabilities=3_641_975_309,
        ),
        CompanyFinancials(
            letter="E", name="中天金融", sector="金融服务", stock_code="300005",
            auditor="致同会计师事务所", period=2023, prev_period=2022,
            revenue=5_678_901_234, cost_of_revenue=4_200_000_000,
            operating_profit=794_446_172, net_profit=624_679_136,
            total_assets=9_876_543_210, current_assets=3_950_617_284,
            total_liabilities=8_641_975_309,
            operating_cash_flow=691_358_025,
            investing_cash_flow=-230_452_675,
            financing_cash_flow=345_679_012,
            prev_revenue=5_432_109_876, prev_net_profit=580_246_914,
            prev_total_assets=9_506_172_840, prev_total_liabilities=8_302_469_136,
        ),
        CompanyFinancials(
            letter="F", name="华信传媒", sector="传媒文化", stock_code="300006",
            auditor="信永中和会计师事务所", period=2023, prev_period=2022,
            revenue=6_789_012_345, cost_of_revenue=4_750_000_000,
            operating_profit=882_572_005, net_profit=712_846_296,
            total_assets=11_111_111_111, current_assets=4_444_444_444,
            total_liabilities=5_555_555_556,
            operating_cash_flow=777_777_778,
            investing_cash_flow=-259_259_259,
            financing_cash_flow=388_888_889,
            prev_revenue=6_543_210_987, prev_net_profit=676_543_210,
            prev_total_assets=10_648_148_148, prev_total_liabilities=5_324_074_074,
        ),
    ],
    "calibration": [
        CompanyFinancials(
            letter="G", name="京东物流", sector="物流运输", stock_code="600101",
            auditor="中审众环会计师事务所", period=2022, prev_period=2021,
            revenue=7_890_123_456, cost_of_revenue=5_920_000_000,
            operating_profit=947_214_815, net_profit=765_341_975,
            total_assets=13_580_246_914, current_assets=5_432_098_766,
            total_liabilities=7_544_080_000,
            operating_cash_flow=832_101_234,
            investing_cash_flow=-277_367_078,
            financing_cash_flow=416_050_617,
            prev_revenue=7_654_321_098, prev_net_profit=728_160_494,
            prev_total_assets=13_179_012_346, prev_total_liabilities=7_320_987_654,
        ),
        CompanyFinancials(
            letter="H", name="顺达运输", sector="交通运输", stock_code="600102",
            auditor="大信会计师事务所", period=2022, prev_period=2021,
            revenue=8_901_234_567, cost_of_revenue=6_680_000_000,
            operating_profit=1_068_148_148, net_profit=890_123_457,
            total_assets=15_432_098_765, current_assets=6_172_839_506,
            total_liabilities=8_487_654_321,
            operating_cash_flow=968_518_519,
            investing_cash_flow=-322_839_506,
            financing_cash_flow=484_259_260,
            prev_revenue=8_765_432_109, prev_net_profit=854_938_272,
            prev_total_assets=15_000_000_000, prev_total_liabilities=8_250_000_000,
        ),
        CompanyFinancials(
            letter="I", name="宏图建筑", sector="建筑工程", stock_code="600103",
            auditor="中汇会计师事务所", period=2022, prev_period=2021,
            revenue=9_012_345_678, cost_of_revenue=8_110_000_000,
            operating_profit=721_000_000, net_profit=540_740_741,
            total_assets=16_222_222_223, current_assets=6_488_888_889,
            total_liabilities=12_166_666_667,
            operating_cash_flow=581_000_000,
            investing_cash_flow=-193_666_667,
            financing_cash_flow=290_500_000,
            prev_revenue=8_876_543_210, prev_net_profit=519_135_802,
            prev_total_assets=15_888_888_889, prev_total_liabilities=11_916_666_667,
        ),
        CompanyFinancials(
            letter="J", name="远大地产", sector="房地产", stock_code="600104",
            auditor="中天运会计师事务所", period=2022, prev_period=2021,
            revenue=10_123_456_789, cost_of_revenue=7_590_000_000,
            operating_profit=1_214_814_815, net_profit=910_111_111,
            total_assets=25_308_641_975, current_assets=10_123_456_790,
            total_liabilities=17_716_049_383,
            operating_cash_flow=1_010_000_000,
            investing_cash_flow=-336_666_667,
            financing_cash_flow=505_000_000,
            prev_revenue=9_987_654_321, prev_net_profit=872_222_222,
            prev_total_assets=24_000_000_000, prev_total_liabilities=16_800_000_000,
        ),
        CompanyFinancials(
            letter="K", name="金辉矿业", sector="矿业开采", stock_code="600105",
            auditor="华兴会计师事务所", period=2022, prev_period=2021,
            revenue=11_234_567_890, cost_of_revenue=7_860_000_000,
            operating_profit=1_680_000_000, net_profit=1_346_000_000,
            total_assets=18_720_000_000, current_assets=7_488_000_000,
            total_liabilities=9_360_000_000,
            operating_cash_flow=1_440_000_000,
            investing_cash_flow=-480_000_000,
            financing_cash_flow=720_000_000,
            prev_revenue=10_976_543_210, prev_net_profit=1_289_000_000,
            prev_total_assets=18_000_000_000, prev_total_liabilities=9_000_000_000,
        ),
        CompanyFinancials(
            letter="L", name="银泰零售", sector="商业零售", stock_code="600106",
            auditor="中兴华会计师事务所", period=2022, prev_period=2021,
            revenue=12_345_678_901, cost_of_revenue=9_870_000_000,
            operating_profit=1_481_481_469, net_profit=1_111_111_101,
            total_assets=20_000_000_000, current_assets=8_000_000_000,
            total_liabilities=10_500_000_000,
            operating_cash_flow=1_200_000_000,
            investing_cash_flow=-400_000_000,
            financing_cash_flow=600_000_000,
            prev_revenue=12_098_765_432, prev_net_profit=1_064_197_531,
            prev_total_assets=19_500_000_000, prev_total_liabilities=10_200_000_000,
        ),
    ],
    "sealed": [
        CompanyFinancials(
            letter="M", name="新世纪电子", sector="电子制造", stock_code="000201",
            auditor="中诚信会计师事务所", period=2024, prev_period=2023,
            revenue=13_456_789_012, cost_of_revenue=9_420_000_000,
            operating_profit=1_750_000_000, net_profit=1_400_000_000,
            total_assets=22_000_000_000, current_assets=8_800_000_000,
            total_liabilities=11_000_000_000,
            operating_cash_flow=1_540_000_000,
            investing_cash_flow=-513_333_333,
            financing_cash_flow=770_000_000,
            prev_revenue=12_987_654_321, prev_net_profit=1_328_395_062,
            prev_total_assets=21_000_000_000, prev_total_liabilities=10_500_000_000,
        ),
        CompanyFinancials(
            letter="N", name="海洋食品", sector="食品加工", stock_code="000202",
            auditor="大华会计师事务所", period=2024, prev_period=2023,
            revenue=14_567_890_123, cost_of_revenue=10_200_000_000,
            operating_profit=1_900_000_000, net_profit=1_580_000_000,
            total_assets=24_000_000_000, current_assets=9_600_000_000,
            total_liabilities=12_500_000_000,
            operating_cash_flow=1_700_000_000,
            investing_cash_flow=-566_666_667,
            financing_cash_flow=850_000_000,
            prev_revenue=14_098_765_432, prev_net_profit=1_504_938_272,
            prev_total_assets=23_000_000_000, prev_total_liabilities=12_000_000_000,
        ),
        CompanyFinancials(
            letter="O", name="天宇航空", sector="航空运输", stock_code="000203",
            auditor="立信会计师事务所", period=2024, prev_period=2023,
            revenue=15_678_901_234, cost_of_revenue=11_760_000_000,
            operating_profit=1_560_000_000, net_profit=1_250_000_000,
            total_assets=35_000_000_000, current_assets=8_750_000_000,
            total_liabilities=22_000_000_000,
            operating_cash_flow=1_900_000_000,
            investing_cash_flow=-633_333_333,
            financing_cash_flow=950_000_000,
            prev_revenue=15_000_000_000, prev_net_profit=1_180_000_000,
            prev_total_assets=33_000_000_000, prev_total_liabilities=21_000_000_000,
        ),
        CompanyFinancials(
            letter="P", name="大地农业", sector="农业种植", stock_code="000204",
            auditor="天健会计师事务所", period=2024, prev_period=2023,
            revenue=16_789_012_345, cost_of_revenue=12_600_000_000,
            operating_profit=1_850_000_000, net_profit=1_600_000_000,
            total_assets=26_000_000_000, current_assets=10_400_000_000,
            total_liabilities=13_000_000_000,
            operating_cash_flow=1_750_000_000,
            investing_cash_flow=-583_333_333,
            financing_cash_flow=875_000_000,
            prev_revenue=16_234_567_890, prev_net_profit=1_528_395_062,
            prev_total_assets=25_000_000_000, prev_total_liabilities=12_500_000_000,
        ),
        CompanyFinancials(
            letter="Q", name="星光教育", sector="教育服务", stock_code="000205",
            auditor="致同会计师事务所", period=2024, prev_period=2023,
            revenue=17_890_123_456, cost_of_revenue=10_730_000_000,
            operating_profit=2_500_000_000, net_profit=2_150_000_000,
            total_assets=28_000_000_000, current_assets=11_200_000_000,
            total_liabilities=11_200_000_000,
            operating_cash_flow=2_300_000_000,
            investing_cash_flow=-766_666_667,
            financing_cash_flow=1_150_000_000,
            prev_revenue=17_234_567_890, prev_net_profit=2_035_802_469,
            prev_total_assets=27_000_000_000, prev_total_liabilities=10_800_000_000,
        ),
        CompanyFinancials(
            letter="R", name="万通通讯", sector="通信服务", stock_code="000206",
            auditor="信永中和会计师事务所", period=2024, prev_period=2023,
            revenue=18_901_234_567, cost_of_revenue=11_340_000_000,
            operating_profit=2_840_000_000, net_profit=2_460_000_000,
            total_assets=30_000_000_000, current_assets=12_000_000_000,
            total_liabilities=15_000_000_000,
            operating_cash_flow=2_660_000_000,
            investing_cash_flow=-886_666_667,
            financing_cash_flow=1_330_000_000,
            prev_revenue=18_234_567_890, prev_net_profit=2_356_790_123,
            prev_total_assets=29_000_000_000, prev_total_liabilities=14_500_000_000,
        ),
    ],
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def format_yuan(n: int) -> str:
    """Format an integer as a comma-separated string (e.g. 1,234,567)."""
    return f"{n:,}"


def to_wan_yuan(n: int) -> str:
    """Convert 元 to 万元 (ten-thousand yuan) with 2 decimal places."""
    return f"{n / 10_000:,.2f} 万元"


def to_baiwan_yuan(n: int) -> str:
    """Convert 元 to 百万元 (million yuan) with 2 decimal places."""
    return f"{n / 1_000_000:,.2f} 百万元"


def compute_sha256(content: str) -> str:
    """Compute the SHA-256 hex digest of a UTF-8 encoded string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------
def build_page1(c: CompanyFinancials) -> str:
    """Build page 1: company overview / front matter.

    Includes company name, stock code, sector, report title, report period,
    comparison period, and auditor.
    """
    return f"""--- Page 1 ---

# {c.name}{c.period}年年度报告

## 公司概况

| 项目 | 内容 |
|------|------|
| 公司名称 | {c.name} |
| 股票代码 | {c.stock_code} |
| 所属行业 | {c.sector} |
| 报告标题 | {c.name}{c.period}年年度报告 |
| 报告期间 | {c.period}年1月1日至{c.period}年12月31日 |
| 上一报告期 | {c.prev_period}年1月1日至{c.prev_period}年12月31日 |
| 审计机构 | {c.auditor} |
| 审计意见 | 标准无保留意见 |

## 公司简介

{c.name}（股票代码：{c.stock_code}，以下简称"本公司"）是一家专注于{c.sector}领域的企业，主要从事{c.sector}相关产品的研发、生产与销售。本公司{c.period}年度报告涵盖{c.period}年1月1日至{c.period}年12月31日的财务状况、经营成果及现金流量情况。

本报告中的财务数据以人民币元为记账本位币，部分汇总数据以万元或百万元为单位列示。除非特别说明，本报告中涉及的金额单位均为元。

## 报告编制基础

本财务报告根据《企业会计准则》编制。本公司{c.period}年度营业收入为{format_yuan(c.revenue)}元（约合{to_wan_yuan(c.revenue)}），{c.prev_period}年度营业收入为{format_yuan(c.prev_revenue)}元（约合{to_wan_yuan(c.prev_revenue)}）。本公司{c.period}年度归属于母公司股东的净利润为{format_yuan(c.net_profit)}元（约合{to_baiwan_yuan(c.net_profit)}）。
"""


def build_page2(c: CompanyFinancials) -> str:
    """Build page 2: income statement.

    Includes revenue, cost, gross profit, operating profit, net profit for
    both periods, plus a financial metrics table with margins and growth.
    """
    return f"""--- Page 2 ---

## 利润表

本页面列示{c.name}{c.period}年度及{c.prev_period}年度的利润表数据，所有金额单位为元。

### 利润表项目

| 项目 | {c.period}年 | {c.prev_period}年 | 单位 |
|------|-------------|-------------------|------|
| 营业收入 | {format_yuan(c.revenue)} | {format_yuan(c.prev_revenue)} | 元 |
| 营业成本 | {format_yuan(c.cost_of_revenue)} | — | 元 |
| 毛利润 | {format_yuan(c.gross_profit)} | — | 元 |
| 营业利润 | {format_yuan(c.operating_profit)} | — | 元 |
| 净利润 | {format_yuan(c.net_profit)} | {format_yuan(c.prev_net_profit)} | 元 |

### 财务指标表

| 指标 | {c.period}年 | 单位 |
|------|-------------|------|
| 毛利率 | {c.gross_margin:.2f} | % |
| 净利率 | {c.net_margin:.2f} | % |
| 营业收入增长率 | {c.revenue_growth:.2f} | % |

## 经营情况分析

{c.period}年度，本公司实现营业收入{format_yuan(c.revenue)}元，较{c.prev_period}年度的{format_yuan(c.prev_revenue)}元增长{c.revenue_growth:.2f}%。营业成本为{format_yuan(c.cost_of_revenue)}元，毛利润为{format_yuan(c.gross_profit)}元，毛利率为{c.gross_margin:.2f}%。

本公司{c.period}年度实现营业利润{format_yuan(c.operating_profit)}元（约合{to_baiwan_yuan(c.operating_profit)}），净利润{format_yuan(c.net_profit)}元（约合{to_baiwan_yuan(c.net_profit)}），净利率为{c.net_margin:.2f}%。{c.prev_period}年度净利润为{format_yuan(c.prev_net_profit)}元。
"""


def build_page3(c: CompanyFinancials) -> str:
    """Build page 3: balance sheet.

    Includes total assets, current assets, total liabilities, equity, and
    debt ratio for both periods.
    """
    return f"""--- Page 3 ---

## 资产负债表

本页面列示{c.name}{c.period}年度及{c.prev_period}年度的资产负债表数据，所有金额单位为元。

### 资产负债表项目

| 项目 | {c.period}年 | {c.prev_period}年 | 单位 |
|------|-------------|-------------------|------|
| 资产总计 | {format_yuan(c.total_assets)} | {format_yuan(c.prev_total_assets)} | 元 |
| 流动资产 | {format_yuan(c.current_assets)} | — | 元 |
| 负债合计 | {format_yuan(c.total_liabilities)} | {format_yuan(c.prev_total_liabilities)} | 元 |
| 所有者权益 | {format_yuan(c.equity)} | — | 元 |
| 资产负债率 | {c.debt_ratio:.2f} | — | % |

## 财务状况分析

截至{c.period}年12月31日，本公司资产总计为{format_yuan(c.total_assets)}元（约合{to_wan_yuan(c.total_assets)}），其中流动资产为{format_yuan(c.current_assets)}元。负债合计为{format_yuan(c.total_liabilities)}元（约合{to_baiwan_yuan(c.total_liabilities)}），所有者权益为{format_yuan(c.equity)}元。

本公司{c.period}年度资产负债率为{c.debt_ratio:.2f}%。{c.prev_period}年度资产总计为{format_yuan(c.prev_total_assets)}元（约合{to_baiwan_yuan(c.prev_total_assets)}），负债合计为{format_yuan(c.prev_total_liabilities)}元。
"""


def build_page4(c: CompanyFinancials) -> str:
    """Build page 4: cash flow statement.

    Includes operating, investing, and financing cash flow for the primary
    period, plus revenue growth rate.
    """
    return f"""--- Page 4 ---

## 现金流量表

本页面列示{c.name}{c.period}年度的现金流量表数据，所有金额单位为元。

### 现金流量表项目

| 项目 | {c.period}年 | 单位 |
|------|-------------|------|
| 经营活动产生的现金流量净额 | {format_yuan(c.operating_cash_flow)} | 元 |
| 投资活动产生的现金流量净额 | {format_yuan(c.investing_cash_flow)} | 元 |
| 筹资活动产生的现金流量净额 | {format_yuan(c.financing_cash_flow)} | 元 |

### 增长率指标

| 指标 | {c.period}年 | 单位 |
|------|-------------|------|
| 营业收入增长率 | {c.revenue_growth:.2f} | % |

## 现金流量分析

{c.period}年度，本公司经营活动产生的现金流量净额为{format_yuan(c.operating_cash_flow)}元（约合{to_wan_yuan(c.operating_cash_flow)}），投资活动产生的现金流量净额为{format_yuan(c.investing_cash_flow)}元，筹资活动产生的现金流量净额为{format_yuan(c.financing_cash_flow)}元（约合{to_baiwan_yuan(c.financing_cash_flow)}）。

本公司{c.period}年度营业收入较{c.prev_period}年度增长{c.revenue_growth:.2f}%，经营性现金流量状况良好。
"""


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------
def build_document(c: CompanyFinancials) -> str:
    """Assemble the complete 4-page Markdown document for a company."""
    pages = [
        build_page1(c),
        build_page2(c),
        build_page3(c),
        build_page4(c),
    ]
    return "\n\n".join(pages) + "\n"


# ---------------------------------------------------------------------------
# Uniqueness verification
# ---------------------------------------------------------------------------
def _metric_tuples(c: CompanyFinancials) -> set[tuple[str, int, str]]:
    """Collect all (metric, period, value) tuples for one company.

    Primary-period metrics include raw and derived figures. Comparison-period
    metrics are limited to the four figures available for the prior year.
    """
    tuples: set[tuple[str, int, str]] = set()
    # Primary period — raw integer metrics
    for metric, value in [
        ("revenue", c.revenue),
        ("cost_of_revenue", c.cost_of_revenue),
        ("gross_profit", c.gross_profit),
        ("operating_profit", c.operating_profit),
        ("net_profit", c.net_profit),
        ("total_assets", c.total_assets),
        ("current_assets", c.current_assets),
        ("total_liabilities", c.total_liabilities),
        ("equity", c.equity),
        ("operating_cash_flow", c.operating_cash_flow),
        ("investing_cash_flow", c.investing_cash_flow),
        ("financing_cash_flow", c.financing_cash_flow),
    ]:
        tuples.add((metric, c.period, str(value)))
    # Primary period — derived percentage metrics
    for metric, value in [
        ("gross_margin", round(c.gross_margin, 2)),
        ("net_margin", round(c.net_margin, 2)),
        ("debt_ratio", round(c.debt_ratio, 2)),
        ("revenue_growth", round(c.revenue_growth, 2)),
    ]:
        tuples.add((metric, c.period, str(value)))
    # Comparison period — only the four available metrics
    for metric, value in [
        ("revenue", c.prev_revenue),
        ("net_profit", c.prev_net_profit),
        ("total_assets", c.prev_total_assets),
        ("total_liabilities", c.prev_total_liabilities),
    ]:
        tuples.add((metric, c.prev_period, str(value)))
    return tuples


def verify_uniqueness(
    companies: dict[str, list[CompanyFinancials]],
) -> list[str]:
    """Verify no (metric, period, value) overlap across partitions.

    Returns a list of error strings. An empty list means all partitions
    are disjoint.
    """
    partition_tuples: dict[str, set[tuple[str, int, str]]] = {}
    for partition, comps in companies.items():
        tuples: set[tuple[str, int, str]] = set()
        for c in comps:
            tuples |= _metric_tuples(c)
        partition_tuples[partition] = tuples

    errors: list[str] = []
    for p1, p2 in itertools.combinations(sorted(partition_tuples), 2):
        overlap = partition_tuples[p1] & partition_tuples[p2]
        if overlap:
            for item in sorted(overlap):
                errors.append(
                    f"(metric, period, value) overlap "
                    f"between {p1} and {p2}: {item}"
                )
    return errors


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------
def write_file(path: Path, content: str) -> None:
    """Write *content* to *path*, overwriting if it already exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_manifest(path: Path, data: dict) -> None:
    """Write the manifest as deterministic pretty JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(payload + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    """Generate the Phase 5 evaluation corpus.

    Returns 0 on success and 1 on any unrecoverable failure.
    """
    print("Generating Phase 5 evaluation corpus...")
    print(f"  output dir : {CORPUS_DIR}")
    print()

    # --- Fail Fast: verify uniqueness before writing any files ---
    print("  verifying (metric, period, value) uniqueness across partitions...")
    errors = verify_uniqueness(COMPANY_DATA)
    if errors:
        for err in errors:
            print(f"  ERROR: {err}", file=sys.stderr)
        print(
            f"FAIL: {len(errors)} uniqueness violation(s) detected.",
            file=sys.stderr,
        )
        return 1
    print("  uniqueness verified — no cross-partition overlap.")
    print()

    # --- Generate all documents and manifest entries ---
    manifest_docs: list[dict] = []
    total_written = 0

    for partition in PARTITIONS:
        companies = COMPANY_DATA[partition]
        part_dir = CORPUS_DIR / partition
        print(f"  [{partition}] {len(companies)} document(s):")
        for c in companies:
            content = build_document(c)
            file_path = part_dir / c.filename
            write_file(file_path, content)
            sha256 = compute_sha256(content)
            size = file_path.stat().st_size
            print(
                f"    [OK] {partition}/{c.filename}  "
                f"({size} bytes, sha256={sha256[:12]}...)"
            )
            manifest_docs.append({
                "filename": c.filename,
                "partition": partition,
                "company": c.name,
                "period": c.period,
                "sha256": sha256,
                "page_count": PAGE_COUNT,
                "source": SOURCE,
                "license": LICENSE,
                "redistribution_allowed": REDISTRIBUTION_ALLOWED,
                "downloaded_at": DOWNLOADED_AT,
            })
            total_written += 1
        print()

    # --- Write manifest ---
    manifest = {
        "corpus": "phase5",
        "generated_at": DOWNLOADED_AT,
        "document_count": len(manifest_docs),
        "documents": manifest_docs,
    }
    manifest_path = CORPUS_DIR / "corpus-manifest.json"
    write_manifest(manifest_path, manifest)
    manifest_size = manifest_path.stat().st_size
    print(
        f"  [OK] corpus-manifest.json  "
        f"({manifest_size} bytes, {len(manifest_docs)} documents)"
    )
    print()

    # --- Summary ---
    print("=" * 60)
    print(f"Phase 5 evaluation corpus generated successfully.")
    print(f"  total documents : {total_written}")
    for partition in PARTITIONS:
        count = len(COMPANY_DATA[partition])
        print(f"    {partition:<14s}: {count} document(s)")
    print(f"  manifest        : {manifest_path.relative_to(ROOT_DIR)}")
    print(f"  uniqueness      : verified (no cross-partition overlap)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
