"""
Script principal: roda diagnósticos para os 5 ETFs e gera tabelas/figura.
Saídas em ../artigos/.
"""

using DataFrames, CSV, Statistics, StatsBase, Random, Printf, Distributions
using Plots
include("analysis.jl")
using .Analysis

const TICKERS = ["SPY", "TLT", "EFA", "EEM", "EWZ"]
const DATA_DIR = joinpath(@__DIR__, "..", "data")
const OUT_DIR = joinpath(@__DIR__, "..", "artigos")
const KS = [1, 5, 10, 21, 42, 63, 126, 252]

# ---------- 1. Carregar dados -------------------------------------------------

dfs = Dict{String, DataFrame}()
for t in TICKERS
    path = joinpath(DATA_DIR, "$(lowercase(t)).csv")
    dfs[t] = load_etf(path)
    @info "$t: $(nrow(dfs[t])) dias úteis, $(dfs[t].date[1])$(dfs[t].date[end])"
end

# ---------- 2. Tabela: amostra & resumo ---------------------------------------

println("\n=== Resumo das séries ===")
df_summary = DataFrame(
    Ticker = TICKERS,
    Início = [string(dfs[t].date[1]) for t in TICKERS],
    Fim = [string(dfs[t].date[end]) for t in TICKERS],
    N = [nrow(dfs[t]) for t in TICKERS],
    σ_anual = [round(std(dfs[t].r) * sqrt(252) * 100, digits=2) for t in TICKERS],
    Curtose = [round(StatsBase.kurtosis(dfs[t].r), digits=2) for t in TICKERS],
    Assimetria = [round(StatsBase.skewness(dfs[t].r), digits=2) for t in TICKERS],
)
CSV.write(joinpath(OUT_DIR, "_tab_summary.csv"), df_summary)
show(df_summary, allcols=true)
println()

# ---------- 3. Tabela: ACF em lags-chave (3 séries × 5 ETFs) ------------------

acf_lags = [1, 5, 21, 63]
acf_rows = DataFrame(
    Ticker = String[], Série = String[],
    L1 = Float64[], L5 = Float64[], L21 = Float64[], L63 = Float64[],
)
for t in TICKERS, (label, series) in [("r", dfs[t].r), ("|r|", dfs[t].abs_r), ("r²", dfs[t].r2)]
    push!(acf_rows, (t, label, [acf_at(series, h) for h in acf_lags]...))
end
CSV.write(joinpath(OUT_DIR, "_tab_acf.csv"), acf_rows)
println("\n=== ACF em lags 1, 5, 21, 63 ===")
show(acf_rows, allcols=true); println()

# ---------- 4. Tabela: Ljung-Box e ARCH-LM ------------------------------------

println("\n=== Ljung-Box (m=20) e ARCH-LM (q=5, q=22) ===")
lb_rows = DataFrame(
    Ticker = String[],
    LB_r_Q = Float64[], LB_r_p = Float64[],
    LB_absr_Q = Float64[], LB_absr_p = Float64[],
    ARCH5_LM = Float64[], ARCH5_p = Float64[],
    ARCH22_LM = Float64[], ARCH22_p = Float64[],
)
for t in TICKERS
    lb_r = ljung_box(dfs[t].r, 20)
    lb_absr = ljung_box(dfs[t].abs_r, 20)
    a5 = arch_lm(dfs[t].r, 5)
    a22 = arch_lm(dfs[t].r, 22)
    push!(lb_rows, (t, lb_r.Q, lb_r.p, lb_absr.Q, lb_absr.p, a5.LM, a5.p, a22.LM, a22.p))
end
CSV.write(joinpath(OUT_DIR, "_tab_lb_arch.csv"), lb_rows)
show(lb_rows, allcols=true); println()

# ---------- 5. Tabela: half-life de log(RV21) ---------------------------------

println("\n=== Half-life log(RV21) e soma parcial autocov ===")
hl_rows = DataFrame(
    Ticker = String[], φ_logRV = Float64[], HL_dias = Float64[],
    PartSum_r_H21 = Float64[], PartSum_absr_H21 = Float64[],
    PartSum_r_H252 = Float64[], PartSum_absr_H252 = Float64[],
)
for t in TICKERS
    hl = halflife_logrv(dfs[t].rv21)
    ps_r_21 = autocov_partial_sum(dfs[t].r, 21)
    ps_absr_21 = autocov_partial_sum(dfs[t].abs_r, 21)
    ps_r_252 = autocov_partial_sum(dfs[t].r, 252)
    ps_absr_252 = autocov_partial_sum(dfs[t].abs_r, 252)
    push!(hl_rows, (t, hl.φ, hl.halflife_days, ps_r_21, ps_absr_21, ps_r_252, ps_absr_252))
end
CSV.write(joinpath(OUT_DIR, "_tab_halflife.csv"), hl_rows)
show(hl_rows, allcols=true); println()

# ---------- 6. Quase-coeficiente m̂(k) ---------------------------------------

println("\n=== m̂(k) por gap (Q=4 quartis) ===")
function mhat_curve_for(x::AbstractVector{<:Real}, ks::Vector{Int})
    return [mhat(x, k; Q=4) for k in ks]
end

# Banda de independência: bootstrap iid (permutação) — só para SPY (custo baixo)
Random.seed!(2026)
spy_r = dfs["SPY"].r
B = 100
mhat_perm_r = [mhat_curve_for(shuffle(spy_r), KS) for _ in 1:B]
mhat_perm_r_mat = reduce(hcat, mhat_perm_r)
band_q95 = [quantile(mhat_perm_r_mat[i, :], 0.95) for i in 1:length(KS)]
band_q50 = [quantile(mhat_perm_r_mat[i, :], 0.50) for i in 1:length(KS)]

mhat_rows = DataFrame(k = KS)
for t in TICKERS
    mhat_rows[!, "$(t)_r"] = mhat_curve_for(dfs[t].r, KS)
    mhat_rows[!, "$(t)_absr"] = mhat_curve_for(dfs[t].abs_r, KS)
    mhat_rows[!, "$(t)_RV21"] = mhat_curve_for(filter(isfinite, dfs[t].rv21), KS)
end
mhat_rows[!, "BandIID_q50_SPY_r"] = band_q50
mhat_rows[!, "BandIID_q95_SPY_r"] = band_q95
CSV.write(joinpath(OUT_DIR, "_tab_mhat.csv"), mhat_rows)
show(mhat_rows, allcols=true); println()

# ---------- 7. Figura central: m̂(k) vs gap ----------------------------------

println("\n=== Figura central: m̂(k) ===")
gr()  # backend Plots
default(fontfamily="Helvetica", titlefontsize=11, guidefontsize=10, tickfontsize=9)

# Subplot por ETF, mas com estilo simples; foco no SPY como painel principal
function mhat_panel(t::String)
    x_r   = mhat_curve_for(dfs[t].r,     KS)
    x_abs = mhat_curve_for(dfs[t].abs_r, KS)
    x_rv  = mhat_curve_for(filter(isfinite, dfs[t].rv21), KS)
    p = plot(KS, x_r, marker=:circle, label="r (retornos)",
             xscale=:log10, lw=2, color=:steelblue,
             xlabel="gap k (dias)", ylabel="m̂(k)", legend=:topright,
             title=t, ylim=(0, max(0.06, 1.05*maximum([maximum(x_r), maximum(x_abs), maximum(x_rv)]))))
    plot!(p, KS, x_abs, marker=:square, label="|r|", lw=2, color=:darkorange)
    plot!(p, KS, x_rv,  marker=:diamond, label="RV21", lw=2, color=:firebrick)
    plot!(p, KS, band_q95, ls=:dash, label="banda iid 95% (SPY)", color=:gray, lw=1)
    return p
end

panels = [mhat_panel(t) for t in TICKERS]
fig = plot(panels..., layout=(2,3), size=(1500, 850), plot_title="")
fig_path_png = joinpath(OUT_DIR, "fig_mhat_panel.png")
savefig(fig, fig_path_png)
@info "Figura salva em $fig_path_png"

# Versão SPY isolado, em alta resolução, para uso de capa
fig_spy = mhat_panel("SPY")
plot!(fig_spy, size=(1100, 700), legendfontsize=10, titlefontsize=14,
      title="Quase-coeficiente de mixing empírico m̂(k) — SPY")
savefig(fig_spy, joinpath(OUT_DIR, "fig_mhat_spy.png"))

println("\nPipeline concluído.")
