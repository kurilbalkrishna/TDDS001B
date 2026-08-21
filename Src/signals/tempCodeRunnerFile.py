buy_hold_return = (df_clean["Close"].iloc[-1] / df_clean["Close"].iloc[len(X_train)] - 1)
print(f"\nBuy-and-Hold Return over Test Period: {buy_hold_return:.4f} ({buy_hold_return*100:.2f}%)")

test_returns = df_clean["Return"].iloc[len(X_train):]
buy_hold_sharpe = (test_returns.mean() / test_returns.std()) * (252 ** 0.5)
print(f"Buy-and-Hold Sharpe Ratio (annualized): {buy_hold_sharpe:.4f}")

from scipy import stats
t_stat, p_value = stats.ttest_rel(reg_scores, rf_scores)
print(f"\nPaired t-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")