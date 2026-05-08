"""
Figura central refinada: m̂(k) para SPY com bandas iid próprias para cada série.
Marca a região k<21 da RV21 como "sobreposição mecânica".
"""

using DataFrames, CSV, Statistics, StatsBase, Random, Printf
using Plots
include("analysis.jl")
using .Analysis

const KS = [1, 5, 10, 21, 42, 63, 126, 252]
const DATA_DIR = joinpath(@__DIR__, "..", "data")
const OUT_DIR = joinpath(@__DIR__, "..", "artigos")

df = load_etf(joinpath(DATA_DIR, "spy.csv"))

Random.seed!(2026)
B = 200

function band_for(x::AbstractVector{<:Real})
    n = length(x)
    samp = zeros(B, length(KS))
    for b in 1:B
        xperm = x[shuffle(1:n)]
        for (i, k) in enumerate(KS)
            samp[b, i] = mhat(xperm, k; Q=4)
        end
    end
    return [quantile(samp[:, i], 0.95) for i in 1:length(KS)]
end

mhat_r   = [mhat(df.r,     k; Q=4) for k in KS]
mhat_abs = [mhat(df.abs_r, k; Q=4) for k in KS]
rv = collect(skipmissing(df.rv21)); rv = rv[isfinite.(rv)]
mhat_rv  = [mhat(rv,       k; Q=4) for k in KS]

band_r   = band_for(df.r)
band_abs = band_for(df.abs_r)
band_rv  = band_for(rv)

println("Banda iid 95%:")
for (i, k) in enumerate(KS)
    @printf("  k=%3d:  r=%.4f   |r|=%.4f   RV21=%.4f\n", k, band_r[i], band_abs[i], band_rv[i])
end

gr()
default(fontfamily="Helvetica", titlefontsize=14, guidefontsize=11, tickfontsize=10,
        legendfontsize=10)

p = plot(KS, mhat_r, marker=:circle, label="r (retornos)",
         xscale=:log10, lw=2.4, color=RGB(0.20, 0.45, 0.75),
         xlabel="gap k entre blocos (dias úteis)",
         ylabel="m̂(k)",
         legend=:topright, size=(1200, 720),
         title="Quase-coeficiente de mixing empírico m̂(k) — SPY (1993–2025, n=8.287)")
plot!(p, KS, mhat_abs, marker=:square, label="|r| (volatilidade lin.)", lw=2.4,
      color=RGB(0.95, 0.55, 0.10))
plot!(p, KS, mhat_rv, marker=:diamond, label="RV21 (volatilidade realizada 21d)", lw=2.4,
      color=RGB(0.78, 0.18, 0.15))

# Bandas iid 95% (média/visual única, pois r e |r| têm bandas muito próximas)
plot!(p, KS, band_r, ls=:dash, label="banda iid 95% (r)", color=RGB(0.20, 0.45, 0.75), lw=1.4, alpha=0.7)
plot!(p, KS, band_abs, ls=:dash, label="banda iid 95% (|r|)", color=RGB(0.95, 0.55, 0.10), lw=1.4, alpha=0.7)
plot!(p, KS, band_rv, ls=:dash, label="banda iid 95% (RV21)", color=RGB(0.78, 0.18, 0.15), lw=1.4, alpha=0.7)

# Sombra para região k < 21 (sobreposição mecânica de RV21)
vspan!(p, [0.9, 21], color=:gray, alpha=0.10, label="")
annotate!(p, 4, 0.165, text("RV21: blocos com\nsobreposição (k<21)", :left, 9, :gray35))

savefig(p, joinpath(OUT_DIR, "fig_mhat_spy.png"))
@info "fig_mhat_spy.png salvo"

# Versão pequena com 5 ETFs em painel
panels = []
tickers = ["SPY", "TLT", "EFA", "EEM", "EWZ"]
for t in tickers
    dfx = load_etf(joinpath(DATA_DIR, "$(lowercase(t)).csv"))
    rvx = collect(skipmissing(dfx.rv21)); rvx = rvx[isfinite.(rvx)]
    mr   = [mhat(dfx.r,     k; Q=4) for k in KS]
    mab  = [mhat(dfx.abs_r, k; Q=4) for k in KS]
    mrv  = [mhat(rvx,       k; Q=4) for k in KS]
    yhi = max(0.06, 1.05*max(maximum(mr), maximum(mab), maximum(mrv)))
    pp = plot(KS, mr, marker=:circle, label="r", xscale=:log10, lw=2,
              color=RGB(0.20, 0.45, 0.75), title=t, ylim=(0, yhi),
              legend=(t=="SPY" ? :topright : false), xlabel="gap k", ylabel="m̂(k)")
    plot!(pp, KS, mab, marker=:square, label="|r|", lw=2, color=RGB(0.95, 0.55, 0.10))
    plot!(pp, KS, mrv, marker=:diamond, label="RV21", lw=2, color=RGB(0.78, 0.18, 0.15))
    push!(panels, pp)
end

fig5 = plot(panels..., layout=(2,3), size=(1500, 900))
savefig(fig5, joinpath(OUT_DIR, "fig_mhat_panel.png"))
@info "fig_mhat_panel.png salvo"
