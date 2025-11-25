def describe_trend_and_solution(df_full: pd.DataFrame, region: str) -> str:
    df_r = df_full[df_full["region"] == region].copy()
    df_r = df_r.sort_values("year")

    # 🔥 데이터 비어있으면 예측 불가 메시지 반환
    if df_r.empty:
        return f"'{region}' 지역에는 예측 데이터가 부족하여 추세 분석을 제공할 수 없습니다."

    x = df_r["year"].values
    y = df_r["pred"].values

    # 🔥 x,y 값이 2개 미만이면 polyfit 불가 → 메시지 반환
    if len(x) < 2:
        return (
            f"'{region}' 지역은 데이터가 매우 적어 추세선을 계산할 수 없습니다.\n"
            "추가 데이터 확보가 필요합니다."
        )
