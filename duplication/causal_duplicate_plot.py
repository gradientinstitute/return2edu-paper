import altair as alt


def correlation_matrix_plot(data):
    correlations = (
        data.corr()
        .stack()
        .reset_index()
        .rename(
            columns={0: "correlation", "level_0": "variable 0", "level_1": "variable 1"}
        )
    )

    # print(correlations.head())
    corr_mat = (
        alt.Chart(correlations)
        .mark_rect()
        .encode(
            x="variable 0:N",
            y="variable 1:N",
            color=alt.Color(
                "correlation:Q",
                scale=alt.Scale(scheme="blueorange", domain=[-1, 1], reverse=True),
            ),
        )
        .properties(width=200, height=200)
    )
    return corr_mat


def distribution_plot(data, feature_name, density=True, counts=True):

    if len(data) >= 5000:
        print("Subsampling 5000 points for plotting")
        plot_data = data.sample(5000)
    else:
        plot_data = data

    base = alt.Chart(plot_data)
    if density:
        dens = (
            base.transform_density(
                feature_name,
                as_=[feature_name, "smooth_counts"],
                counts=counts,  # we likely want counts rather than pdf
            )
            .mark_area()
            .encode(x=f"{feature_name}:Q", y="smooth_counts:Q")
        )
    else:
        dens = base.mark_bar().encode(alt.X(f"{feature_name}:Q", bin=True), y="count()")
    mean = base.mark_rule(color="red").encode(
        x=f"mean({feature_name}):Q",
        # size=alt.value(5)
    )
    return dens + mean
