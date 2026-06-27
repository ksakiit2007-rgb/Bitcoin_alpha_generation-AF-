from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = Path(r"C:\Users\RAKSHIT\Downloads\btc_alpha_master_2014_2026 (1).csv")
OUT_DIR = ROOT / "outputs"
WORK_DIR = ROOT / "work"


FEATURES = [
    "log_True_Market_Cap",
    "log_FeeTotUSD",
    "log_AdrActCnt",
    "log_HashRate",
    "log_Fee_to_MCap",
    "DXY_Close",
    "TNX_Yield",
    "Rolling_Volatility",
]

PRETTY_FEATURES = {
    "log_True_Market_Cap": "True Market Cap",
    "log_FeeTotUSD": "Fees in USD",
    "log_AdrActCnt": "Active addresses",
    "log_HashRate": "Hash rate",
    "log_Fee_to_MCap": "Fee to market cap",
    "DXY_Close": "US Dollar Index",
    "TNX_Yield": "10Y Treasury yield",
    "Rolling_Volatility": "Rolling volatility",
}


def fmt_pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f}%"


def fmt_num(x: float, digits: int = 4) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"


def max_drawdown(simple_returns: np.ndarray) -> float:
    equity = np.cumprod(1 + simple_returns)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1
    return float(np.min(drawdown))


def cumulative_return(simple_returns: np.ndarray) -> float:
    return float(np.prod(1 + simple_returns) - 1)


def sharpe(simple_returns: np.ndarray) -> float:
    sd = np.std(simple_returns, ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(np.sqrt(365) * np.mean(simple_returns) / sd)


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    denom = np.sum((y - np.mean(y)) ** 2)
    if denom == 0:
        return 0.0
    return float(1 - np.sum((y - pred) ** 2) / denom)


def directional_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y) == np.sign(pred)))


def make_signal(pred: np.ndarray, threshold: float = 0.0, orientation: int = 1) -> np.ndarray:
    return orientation * np.where(pred >= threshold, 1, -1)


def strategy_metrics(y_log: np.ndarray, pred: np.ndarray, threshold: float = 0.0, orientation: int = 1) -> dict[str, float]:
    signal = make_signal(pred, threshold, orientation)
    btc_ret = np.exp(y_log) - 1
    strat_ret = signal * btc_ret
    return {
        "cumulative_return": cumulative_return(strat_ret),
        "buy_hold_return": cumulative_return(btc_ret),
        "annualized_sharpe": sharpe(strat_ret),
        "max_drawdown": max_drawdown(strat_ret),
        "long_days": float(np.mean(signal > 0)),
    }


def standardize(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series]:
    mu = train.mean()
    sd = train.std(ddof=0).replace(0, 1)
    return (
        ((train - mu) / sd).to_numpy(),
        ((val - mu) / sd).to_numpy(),
        ((test - mu) / sd).to_numpy(),
        mu,
        sd,
    )


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def fit_ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(add_intercept(x)) @ y


def predict_linear(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return add_intercept(x) @ beta


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x1 = add_intercept(x)
    penalty = np.eye(x1.shape[1]) * alpha
    penalty[0, 0] = 0
    return np.linalg.solve(x1.T @ x1 + penalty, x1.T @ y)


def fit_lasso_cd(x: np.ndarray, y: np.ndarray, alpha: float, iters: int = 2500) -> np.ndarray:
    x1 = add_intercept(x)
    beta = np.zeros(x1.shape[1])
    col_norm = np.sum(x1**2, axis=0)

    def soft_threshold(z: float, gamma: float) -> float:
        if z > gamma:
            return z - gamma
        if z < -gamma:
            return z + gamma
        return 0.0

    for _ in range(iters):
        old = beta.copy()
        for j in range(x1.shape[1]):
            residual = y - x1 @ beta + x1[:, j] * beta[j]
            rho = float(x1[:, j] @ residual)
            if j == 0:
                beta[j] = rho / col_norm[j]
            else:
                beta[j] = soft_threshold(rho, alpha) / col_norm[j]
        if np.max(np.abs(beta - old)) < 1e-10:
            break
    return beta


@dataclass
class Stump:
    feature: int
    threshold: float
    left_value: float
    right_value: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.where(x[:, self.feature] <= self.threshold, self.left_value, self.right_value)


def best_stump(x: np.ndarray, residual: np.ndarray) -> Stump:
    best = Stump(0, 0.0, float(np.mean(residual)), float(np.mean(residual)))
    best_loss = float("inf")
    for j in range(x.shape[1]):
        values = np.unique(np.quantile(x[:, j], np.linspace(0.05, 0.95, 19)))
        for threshold in values:
            left = residual[x[:, j] <= threshold]
            right = residual[x[:, j] > threshold]
            if len(left) < 25 or len(right) < 25:
                continue
            lv = float(np.mean(left))
            rv = float(np.mean(right))
            pred = np.where(x[:, j] <= threshold, lv, rv)
            loss = float(np.mean((residual - pred) ** 2))
            if loss < best_loss:
                best_loss = loss
                best = Stump(j, float(threshold), lv, rv)
    return best


class BoostedStumps:
    def __init__(self, n_estimators: int = 80, learning_rate: float = 0.04):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base = 0.0
        self.stumps: list[Stump] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BoostedStumps":
        self.base = float(np.mean(y))
        pred = np.full(len(y), self.base)
        self.stumps = []
        for _ in range(self.n_estimators):
            residual = y - pred
            stump = best_stump(x, residual)
            pred += self.learning_rate * stump.predict(x)
            self.stumps.append(stump)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        pred = np.full(len(x), self.base)
        for stump in self.stumps:
            pred += self.learning_rate * stump.predict(x)
        return pred

    def feature_importance(self, n_features: int) -> np.ndarray:
        out = np.zeros(n_features)
        for stump in self.stumps:
            out[stump.feature] += abs(stump.left_value - stump.right_value)
        if out.sum() > 0:
            out = out / out.sum()
        return out


def evaluate_model(name: str, y: np.ndarray, pred: np.ndarray, threshold: float = 0.0, orientation: int = 1) -> dict[str, float | str]:
    perf = strategy_metrics(y, pred, threshold, orientation)
    out: dict[str, float | str] = {
        "model": name,
        "MAE": mae(y, pred),
        "RMSE": rmse(y, pred),
        "Directional_Accuracy": directional_accuracy(y, pred),
        "R2": r2(y, pred),
    }
    out.update(perf)
    return out


def calibrate_signal(train_pred: np.ndarray, val_pred: np.ndarray, y_val: np.ndarray) -> tuple[float, int]:
    thresholds = list(np.quantile(train_pred, [0.2, 0.35, 0.5, 0.65, 0.8]))
    thresholds += list(np.quantile(val_pred, [0.2, 0.35, 0.5, 0.65, 0.8]))
    thresholds.append(0.0)
    best_threshold = thresholds[0]
    best_orientation = 1
    best_score = -float("inf")
    for threshold in thresholds:
        for orientation in [1, -1]:
            score = strategy_metrics(y_val, val_pred, float(threshold), orientation)["annualized_sharpe"]
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
                best_orientation = orientation
    return best_threshold, best_orientation


def prepare_data() -> tuple[pd.DataFrame, dict[str, object]]:
    df = pd.read_csv(INPUT_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Target_Next_Log_Return"] = np.log(df["Close"].shift(-1) / df["Close"])
    df["Rolling_Volatility"] = df["Log_Return"].rolling(30).std()

    for col in ["True_Market_Cap", "FeeTotUSD", "AdrActCnt", "HashRate", "Fee_to_MCap"]:
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    clean = df.dropna(subset=FEATURES + ["Target_Next_Log_Return"]).copy()
    n = len(clean)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    clean["Split"] = "Test"
    clean.loc[clean.index[:train_end], "Split"] = "Train"
    clean.loc[clean.index[train_end:val_end], "Split"] = "Validation"

    meta = {
        "raw_rows": len(df),
        "model_rows": n,
        "date_start": clean["Date"].min().strftime("%Y-%m-%d"),
        "date_end": clean["Date"].max().strftime("%Y-%m-%d"),
        "train_period": f"{clean.iloc[0]['Date'].date()} to {clean.iloc[train_end - 1]['Date'].date()}",
        "val_period": f"{clean.iloc[train_end]['Date'].date()} to {clean.iloc[val_end - 1]['Date'].date()}",
        "test_period": f"{clean.iloc[val_end]['Date'].date()} to {clean.iloc[-1]['Date'].date()}",
        "missing_values": int(df.isna().sum().sum()),
    }
    return clean, meta


def market_regime_table(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["Return_90d"] = working["Close"].pct_change(90)
    working["DXY_30d"] = working["DXY_Close"].pct_change(30)
    working["TNX_30d"] = working["TNX_Yield"].diff(30)
    conditions = [
        working["Return_90d"] > 0.30,
        working["Return_90d"] < -0.20,
        (working["DXY_30d"] > 0.03) | (working["TNX_30d"] > 0.50),
    ]
    choices = ["Bull", "Bear", "Macro shock"]
    working["Regime"] = np.select(conditions, choices, default="Consolidation")
    summary = (
        working.dropna(subset=["Return_90d"])
        .groupby("Regime")
        .agg(
            days=("Date", "count"),
            avg_next_return=("Target_Next_Log_Return", "mean"),
            volatility=("Target_Next_Log_Return", "std"),
            avg_dxy=("DXY_Close", "mean"),
            avg_tnx=("TNX_Yield", "mean"),
            avg_active_addresses=("AdrActCnt", "mean"),
        )
        .reset_index()
    )
    return summary


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = FEATURES + ["Target_Next_Log_Return"]
    corr = df[cols].corr()["Target_Next_Log_Return"].drop("Target_Next_Log_Return")
    out = corr.rename("corr_to_next_return").reset_index().rename(columns={"index": "feature"})
    out["feature"] = out["feature"].map(PRETTY_FEATURES)
    return out.sort_values("corr_to_next_return", key=lambda x: np.abs(x), ascending=False)


def train_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    train = df[df["Split"] == "Train"]
    val = df[df["Split"] == "Validation"]
    test = df[df["Split"] == "Test"]
    x_train, x_val, x_test, mu, sd = standardize(train[FEATURES], val[FEATURES], test[FEATURES])
    y_train = train["Target_Next_Log_Return"].to_numpy()
    y_val = val["Target_Next_Log_Return"].to_numpy()
    y_test = test["Target_Next_Log_Return"].to_numpy()

    candidates: dict[str, tuple[object, np.ndarray, np.ndarray, np.ndarray, float, int]] = {}

    baseline_pred_train = np.full(len(y_train), np.mean(y_train))
    baseline_pred_val = np.full(len(y_val), np.mean(y_train))
    baseline_pred_test = np.full(len(y_test), np.mean(y_train))
    base_threshold, base_orientation = calibrate_signal(baseline_pred_train, baseline_pred_val, y_val)
    candidates["Mean baseline"] = ("mean", baseline_pred_train, baseline_pred_val, baseline_pred_test, base_threshold, base_orientation)

    ols = fit_ols(x_train, y_train)
    ols_train = predict_linear(ols, x_train)
    ols_val = predict_linear(ols, x_val)
    threshold, orientation = calibrate_signal(ols_train, ols_val, y_val)
    candidates["Linear regression"] = (ols, ols_train, ols_val, predict_linear(ols, x_test), threshold, orientation)

    for alpha in [0.1, 1.0, 5.0, 20.0, 80.0]:
        model = fit_ridge(x_train, y_train, alpha)
        train_pred = predict_linear(model, x_train)
        val_pred = predict_linear(model, x_val)
        threshold, orientation = calibrate_signal(train_pred, val_pred, y_val)
        candidates[f"Ridge alpha {alpha:g}"] = (model, train_pred, val_pred, predict_linear(model, x_test), threshold, orientation)

    for alpha in [0.0002, 0.0005, 0.001, 0.002]:
        model = fit_lasso_cd(x_train, y_train, alpha)
        train_pred = predict_linear(model, x_train)
        val_pred = predict_linear(model, x_val)
        threshold, orientation = calibrate_signal(train_pred, val_pred, y_val)
        candidates[f"Lasso alpha {alpha:g}"] = (model, train_pred, val_pred, predict_linear(model, x_test), threshold, orientation)

    for n_est, lr in [(40, 0.04), (80, 0.04), (120, 0.03)]:
        model = BoostedStumps(n_est, lr).fit(x_train, y_train)
        train_pred = model.predict(x_train)
        val_pred = model.predict(x_val)
        threshold, orientation = calibrate_signal(train_pred, val_pred, y_val)
        candidates[f"Boosted stumps {n_est}"] = (model, train_pred, val_pred, model.predict(x_test), threshold, orientation)

    val_rows = [evaluate_model(name, y_val, val_pred, threshold, orientation) for name, (_, _, val_pred, _, threshold, orientation) in candidates.items()]
    val_metrics = pd.DataFrame(val_rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    ml_only = val_metrics[val_metrics["model"] != "Mean baseline"].copy()
    ml_only = ml_only.sort_values(["annualized_sharpe", "Directional_Accuracy", "RMSE"], ascending=[False, False, True])
    best_name = str(ml_only.iloc[0]["model"])

    test_rows = [evaluate_model(name, y_test, test_pred, threshold, orientation) for name, (_, _, _, test_pred, threshold, orientation) in candidates.items()]
    test_metrics = pd.DataFrame(test_rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)

    best_model, _, _, best_pred, best_threshold, best_orientation = candidates[best_name]
    feature_imp = feature_importance(best_model, x_test, y_test)

    pred_table = test[["Date", "Close", "Target_Next_Log_Return"]].copy()
    pred_table["Predicted_Next_Log_Return"] = best_pred
    best_signal = make_signal(best_pred, best_threshold, best_orientation)
    pred_table["Signal"] = np.where(best_signal == 1, "Long", "Short")
    pred_table["BTC_Next_Return"] = np.exp(pred_table["Target_Next_Log_Return"]) - 1
    pred_table["Strategy_Return"] = np.where(pred_table["Signal"] == "Long", 1, -1) * pred_table["BTC_Next_Return"]
    pred_table["Strategy_Equity"] = (1 + pred_table["Strategy_Return"]).cumprod()
    pred_table["BTC_Buy_Hold_Equity"] = (1 + pred_table["BTC_Next_Return"]).cumprod()

    info = {
        "best_model": best_name,
        "signal_threshold": best_threshold,
        "signal_orientation": best_orientation,
        "feature_means": mu.to_dict(),
        "feature_stds": sd.to_dict(),
    }
    return val_metrics, test_metrics, pred_table, {"info": info, "feature_importance": feature_imp}


def feature_importance(model: object, x_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    if isinstance(model, BoostedStumps):
        values = model.feature_importance(len(FEATURES))
    elif isinstance(model, np.ndarray):
        values = np.abs(model[1:])
        values = values / values.sum() if values.sum() else values
    else:
        values = np.zeros(len(FEATURES))

    base_pred = model.predict(x_test) if isinstance(model, BoostedStumps) else predict_linear(model, x_test)
    base_rmse = rmse(y_test, base_pred)
    rng = np.random.default_rng(42)
    perm = []
    for j in range(x_test.shape[1]):
        drops = []
        for _ in range(20):
            x_perm = x_test.copy()
            rng.shuffle(x_perm[:, j])
            pred = model.predict(x_perm) if isinstance(model, BoostedStumps) else predict_linear(model, x_perm)
            drops.append(rmse(y_test, pred) - base_rmse)
        perm.append(float(np.mean(drops)))

    out = pd.DataFrame(
        {
            "feature": [PRETTY_FEATURES[f] for f in FEATURES],
            "model_importance": values,
            "permutation_rmse_increase": perm,
        }
    )
    return out.sort_values("permutation_rmse_increase", ascending=False)


class LineChart(Flowable):
    def __init__(self, series: list[tuple[str, list[float], colors.Color]], width: float = 6.8 * inch, height: float = 2.8 * inch):
        super().__init__()
        self.series = series
        self.width = width
        self.height = height

    def draw(self) -> None:
        c = self.canv
        left, bottom = 0.45 * inch, 0.3 * inch
        w, h = self.width - 0.75 * inch, self.height - 0.55 * inch
        all_values = [v for _, vals, _ in self.series for v in vals]
        mn, mx = min(all_values), max(all_values)
        if mx == mn:
            mx = mn + 1
        c.setStrokeColor(colors.HexColor("#9CA3AF"))
        c.line(left, bottom, left, bottom + h)
        c.line(left, bottom, left + w, bottom)

        for name, values, color in self.series:
            c.setStrokeColor(color)
            c.setLineWidth(1.4)
            points = []
            for i, val in enumerate(values):
                x = left + (i / max(1, len(values) - 1)) * w
                y = bottom + ((val - mn) / (mx - mn)) * h
                points.append((x, y))
            for p1, p2 in zip(points, points[1:]):
                c.line(p1[0], p1[1], p2[0], p2[1])

        c.setFont("Helvetica", 8)
        x0 = left
        for name, _, color in self.series:
            c.setFillColor(color)
            c.rect(x0, bottom + h + 8, 8, 8, fill=1, stroke=0)
            c.setFillColor(colors.black)
            c.drawString(x0 + 12, bottom + h + 8, name)
            x0 += 1.7 * inch


def make_pdf_report(
    meta: dict[str, object],
    val_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    corr: pd.DataFrame,
    regimes: pd.DataFrame,
    importance: pd.DataFrame,
    selected_model: str,
    pdf_path: Path,
) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CenterTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11))
    styles.add(ParagraphStyle(name="Plain", parent=styles["BodyText"], fontSize=10.2, leading=14, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontSize=13, leading=16, spaceBefore=10, spaceAfter=5))

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )

    story = []
    story.append(Paragraph("Alpha Forge Bitcoin Alpha Report", styles["CenterTitle"]))
    story.append(Paragraph("On-chain and macro ML modelling", styles["Plain"]))
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("What I did", styles["Section"]))
    story.append(
        Paragraph(
            "I cleaned the daily Bitcoin dataset, built the Rhode feature set, created a next-day log return target, "
            "trained several simple machine learning models, turned the best validation model into a daily long/short signal, "
            "and tested it against Bitcoin buy-and-hold. The split is chronological, so later rows never leak into earlier training.",
            styles["Plain"],
        )
    )

    overview = [
        ["Raw rows", f"{meta['raw_rows']:,}"],
        ["Usable model rows", f"{meta['model_rows']:,}"],
        ["Model date range", f"{meta['date_start']} to {meta['date_end']}"],
        ["Train", str(meta["train_period"])],
        ["Validation", str(meta["val_period"])],
        ["Test", str(meta["test_period"])],
    ]
    story.append(make_table(overview, ["Item", "Value"], [2.0 * inch, 4.6 * inch]))

    best_name = selected_model
    best_test = test_metrics[test_metrics["model"] == best_name].iloc[0]
    story.append(Paragraph("Main result", styles["Section"]))
    story.append(
        Paragraph(
            f"The selected model was <b>{best_name}</b>, chosen on the validation set by the best strategy Sharpe ratio. "
            f"On the untouched test period, it reached directional accuracy of {fmt_pct(best_test['Directional_Accuracy'])}, "
            f"an annualized Sharpe ratio of {fmt_num(best_test['annualized_sharpe'], 2)}, "
            f"and cumulative strategy return of {fmt_pct(best_test['cumulative_return'])}. "
            f"Bitcoin buy-and-hold over the same test rows returned {fmt_pct(best_test['buy_hold_return'])}.",
            styles["Plain"],
        )
    )

    story.append(LineChart([
        ("Strategy", predictions["Strategy_Equity"].tolist(), colors.HexColor("#2563EB")),
        ("BTC buy-hold", predictions["BTC_Buy_Hold_Equity"].tolist(), colors.HexColor("#DC2626")),
    ]))

    story.append(Paragraph("Test metrics", styles["Section"]))
    display = test_metrics.copy()
    display = display[["model", "MAE", "RMSE", "Directional_Accuracy", "R2", "annualized_sharpe", "max_drawdown", "cumulative_return", "buy_hold_return"]]
    rows = []
    for _, row in display.head(8).iterrows():
        rows.append([
            row["model"],
            fmt_num(row["MAE"], 5),
            fmt_num(row["RMSE"], 5),
            fmt_pct(row["Directional_Accuracy"], 1),
            fmt_num(row["R2"], 3),
            fmt_num(row["annualized_sharpe"], 2),
            fmt_pct(row["max_drawdown"], 1),
            fmt_pct(row["cumulative_return"], 1),
            fmt_pct(row["buy_hold_return"], 1),
        ])
    story.append(make_table(rows, ["Model", "MAE", "RMSE", "Dir.", "R2", "Sharpe", "Max DD", "Strategy", "BTC"], [1.35 * inch] + [0.65 * inch] * 8, font_size=7.2))

    story.append(PageBreak())
    story.append(Paragraph("Feature checks", styles["Section"]))
    corr_rows = [[r["feature"], fmt_num(r["corr_to_next_return"], 4)] for _, r in corr.iterrows()]
    story.append(make_table(corr_rows, ["Feature", "Correlation with next-day return"], [2.4 * inch, 2.4 * inch]))
    story.append(
        Paragraph(
            "The raw correlations are weak, which is normal for daily Bitcoin returns. This means the model should be judged more by out-of-sample performance than by any single correlation value.",
            styles["Plain"],
        )
    )

    story.append(Paragraph("Market regimes", styles["Section"]))
    regime_rows = []
    for _, row in regimes.iterrows():
        regime_rows.append([
            row["Regime"],
            f"{int(row['days']):,}",
            fmt_pct(row["avg_next_return"], 3),
            fmt_pct(row["volatility"], 2),
            fmt_num(row["avg_dxy"], 2),
            fmt_num(row["avg_tnx"], 2),
            f"{row['avg_active_addresses']:,.0f}",
        ])
    story.append(make_table(regime_rows, ["Regime", "Days", "Avg next return", "Vol.", "DXY", "TNX", "Active addr."], [1.15 * inch, 0.55 * inch, 0.95 * inch, 0.6 * inch, 0.55 * inch, 0.55 * inch, 0.95 * inch], font_size=7.8))

    story.append(PageBreak())
    story.append(Paragraph("Feature importance", styles["Section"]))
    imp_rows = []
    for _, row in importance.iterrows():
        imp_rows.append([
            row["feature"],
            fmt_num(row["model_importance"], 3),
            fmt_num(row["permutation_rmse_increase"], 6),
        ])
    story.append(make_table(imp_rows, ["Feature", "Model weight", "RMSE increase when shuffled"], [2.2 * inch, 1.3 * inch, 2.0 * inch]))
    top = importance.iloc[0]["feature"]
    story.append(
        Paragraph(
            f"The strongest feature by permutation testing was <b>{top}</b>. "
            "This does not mean it predicts every day by itself. It means the trained model lost the most accuracy when that input was disturbed.",
            styles["Plain"],
        )
    )

    story.append(Paragraph("Data preparation notes", styles["Section"]))
    story.append(
        Paragraph(
            "I used the eight input Rhode features and built 30-day rolling volatility from past log returns. "
            "Large on-chain values were log-transformed, then all model inputs were standardized using only the training period. "
            "The target was next-day log return. This lets the signal be used after today's close without seeing tomorrow's price. "
            "The long/short cutoff was calibrated on the validation period, then held fixed for the test period.",
            styles["Plain"],
        )
    )

    story.append(Paragraph("Conclusion", styles["Section"]))
    story.append(
        Paragraph(
            "The signal is useful as a research alpha, not as a finished trading system. It shows that on-chain activity, network valuation, macro conditions, and volatility contain some information about the next day's return direction. "
            "The main weakness is that daily Bitcoin returns are noisy, so small statistical gains can disappear after trading costs, slippage, or regime changes. "
            "A production version should add transaction costs, position sizing, walk-forward retraining, and stricter risk limits.",
            styles["Plain"],
        )
    )

    doc.build(story)


def make_table(rows: list[list[object]], header: list[str], widths: list[float], font_size: float = 8.5) -> Table:
    data = [header] + rows
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 2),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F9FAFB")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F3F4F6")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    df, meta = prepare_data()
    corr = correlation_table(df)
    regimes = market_regime_table(df)
    val_metrics, test_metrics, predictions, model_info = train_models(df)
    importance = model_info["feature_importance"]

    df.to_csv(OUT_DIR / "alpha_forge_prepared_dataset.csv", index=False)
    predictions.to_csv(OUT_DIR / "alpha_forge_test_predictions.csv", index=False)
    val_metrics.to_csv(OUT_DIR / "alpha_forge_validation_metrics.csv", index=False)
    test_metrics.to_csv(OUT_DIR / "alpha_forge_test_metrics.csv", index=False)
    corr.to_csv(OUT_DIR / "alpha_forge_feature_correlations.csv", index=False)
    regimes.to_csv(OUT_DIR / "alpha_forge_regime_summary.csv", index=False)
    importance.to_csv(OUT_DIR / "alpha_forge_feature_importance.csv", index=False)

    summary = {
        "metadata": meta,
        "selected_model": model_info["info"]["best_model"],
        "selected_model_test_metrics": test_metrics[test_metrics["model"] == model_info["info"]["best_model"]].iloc[0].to_dict(),
    }
    (OUT_DIR / "alpha_forge_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    make_pdf_report(
        meta,
        val_metrics,
        test_metrics,
        predictions,
        corr,
        regimes,
        importance,
        str(model_info["info"]["best_model"]),
        OUT_DIR / "alpha_forge_report.pdf",
    )

    script_copy = OUT_DIR / "alpha_forge_reproducible_analysis.py"
    script_copy.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
