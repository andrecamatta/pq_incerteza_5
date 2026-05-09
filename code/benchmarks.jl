"""
Benchmarks de mixing: GARCH(1,1) (geometricamente β-mixing por Carrasco-Chen 2002)
e Markov-Switching GARCH (proxy de long-memory aparente via mudança lenta de regime,
na linha de Mikosch-Stărică 2004). Gera bandas teóricas de m̂(k) para sobrepor na
figura central do SPY.

Saída: artigos/_tab_benchmarks.csv  (curvas medianas e bandas 5%-95%)
       artigos/fig_mhat_spy_benchmarks.png
"""

using DataFrames, CSV, Statistics, StatsBase, Random, Printf
using Distributions, Optim, LinearAlgebra
using Plots
include("analysis.jl")
using .Analysis

const KS = [1, 5, 10, 21, 42, 63, 126, 252]
const DATA_DIR = joinpath(@__DIR__, "..", "data")
const OUT_DIR  = joinpath(@__DIR__, "..", "artigos")

# ---------- 1. QMLE de GARCH(1,1) em SPY --------------------------------------

df_spy = load_etf(joinpath(DATA_DIR, "spy.csv"))
r_spy = df_spy.r .- mean(df_spy.r)        # tira o drift
T = length(r_spy)

"Negative log-likelihood de GARCH(1,1) Gaussiano."
function nll_garch11(θ::Vector{Float64}, r::Vector{Float64})
    ω, α, β = exp(θ[1]), 1/(1+exp(-θ[2])), 0.0
    # parametrização que garante 0<α<1 e α+β<1
    # θ[3] controla β/(1-α): logit
    γ = 1/(1+exp(-θ[3]))     # γ ∈ (0,1)
    β = γ * (1 - α)          # β = γ·(1-α), garante α+β = α + γ(1-α) ∈ (α, 1)
    n = length(r)
    σ2 = var(r)
    nll = 0.0
    @inbounds for t in 1:n
        nll += 0.5 * (log(2π * σ2) + r[t]^2 / σ2)
        σ2 = ω + α * r[t]^2 + β * σ2
        if !isfinite(σ2) || σ2 ≤ 0
            return 1e10
        end
    end
    return nll
end

θ0 = [log(1e-6), log(0.08/(1-0.08)), log(0.91/0.99 / (1 - 0.91/0.99))]
res = optimize(θ -> nll_garch11(θ, r_spy), θ0, NelderMead(),
               Optim.Options(iterations=2000))
θ̂ = Optim.minimizer(res)
ω_h = exp(θ̂[1])
α_h = 1/(1+exp(-θ̂[2]))
γ_h = 1/(1+exp(-θ̂[3]))
β_h = γ_h * (1 - α_h)

@printf("\nGARCH(1,1) QMLE em SPY:\n  ω = %.3e\n  α = %.4f\n  β = %.4f\n  α+β = %.4f\n  half-life σ² (dias) = %.1f\n",
        ω_h, α_h, β_h, α_h+β_h, log(0.5)/log(α_h+β_h))

# ---------- 2. Simuladores ----------------------------------------------------

"Gera trajetória GARCH(1,1) Gaussiano de tamanho n com parâmetros (ω, α, β)."
function sim_garch11(rng, n::Int, ω::Float64, α::Float64, β::Float64)
    r = zeros(n)
    σ2 = ω / (1 - α - β)        # variância incondicional
    @inbounds for t in 1:n
        ε = randn(rng)
        r[t] = sqrt(σ2) * ε
        σ2 = ω + α * r[t]^2 + β * σ2
    end
    return r
end

"""
Markov-Switching GARCH com 2 regimes:
- Estado 1 (calmo): GARCH(1,1) com ω_1
- Estado 2 (agitado): GARCH(1,1) com ω_2 = κ·ω_1 (κ > 1)
- Cadeia Markov 2-estados com transição (p_cc, p_aa) altamente persistente.

A persistência longa do regime cria mudanças estruturais lentas em variância,
que produzem ACF de |r| com decaimento lento mesmo que cada regime seja
short-memory mixing — proxy para long-memory aparente (MIKOSCH; STĂRICĂ, 2004).
"""
function sim_msgarch(rng, n::Int; ω1=1.0e-6, κ=4.0, α=0.06, β=0.93,
                     p_cc=0.995, p_aa=0.97)
    ω2 = κ * ω1
    state = 1
    r = zeros(n); states = zeros(Int, n)
    σ2 = ω1 / (1 - α - β)
    @inbounds for t in 1:n
        # transição
        u = rand(rng)
        if state == 1
            state = (u > p_cc) ? 2 : 1
        else
            state = (u > p_aa) ? 1 : 2
        end
        states[t] = state
        ω_t = (state == 1) ? ω1 : ω2
        ε = randn(rng)
        r[t] = sqrt(σ2) * ε
        σ2 = ω_t + α * r[t]^2 + β * σ2
    end
    return r, states
end

# ---------- 3. Computar bandas m̂(k) por simulação ----------------------------

"Calcula RV21 a partir de uma série de retornos."
function rv21(r::Vector{Float64})
    n = length(r)
    rv = fill(NaN, n)
    @inbounds for i in 21:n
        rv[i] = sum(view(r, (i-20):i) .^ 2)
    end
    return rv
end

function mhat_curve(x::AbstractVector, ks::Vector{Int})
    return [mhat(x, k; Q=4) for k in ks]
end

"Roda B simulações de tamanho T do gerador `gen` e devolve quantis 5%/50%/95% de m̂(k)."
function sim_band(gen::Function, B::Int, T::Int, ks::Vector{Int}; seed=2026)
    rng = MersenneTwister(seed)
    n = length(ks)
    M_r   = zeros(B, n)
    M_abs = zeros(B, n)
    M_rv  = zeros(B, n)
    for b in 1:B
        r_sim = gen(rng, T)
        rv_sim = rv21(r_sim)
        rv_clean = rv_sim[isfinite.(rv_sim) .& (rv_sim .> 0)]
        M_r[b, :]   .= mhat_curve(r_sim, ks)
        M_abs[b, :] .= mhat_curve(abs.(r_sim), ks)
        M_rv[b, :]  .= mhat_curve(rv_clean, ks)
    end
    Q = (M, q) -> [quantile(M[:, j], q) for j in 1:n]
    return (
        r   = (q05=Q(M_r,0.05), q50=Q(M_r,0.50), q95=Q(M_r,0.95)),
        abs = (q05=Q(M_abs,0.05), q50=Q(M_abs,0.50), q95=Q(M_abs,0.95)),
        rv  = (q05=Q(M_rv,0.05), q50=Q(M_rv,0.50), q95=Q(M_rv,0.95)),
    )
end

@info "Simulando GARCH(1,1) calibrado..."
B = 200
T_sim = T   # mesmo tamanho de SPY
gen_garch = (rng, n) -> sim_garch11(rng, n, ω_h, α_h, β_h)
band_garch = sim_band(gen_garch, B, T_sim, KS; seed=2026)

@info "Simulando MS-GARCH (proxy long-memory aparente)..."
gen_ms = (rng, n) -> sim_msgarch(rng, n; ω1=ω_h, κ=4.0, α=α_h, β=β_h, p_cc=0.995, p_aa=0.97)[1]
band_ms = sim_band(gen_ms, B, T_sim, KS; seed=2026)

# ---------- 4. m̂(k) empírico do SPY ------------------------------------------

mhat_spy_r   = mhat_curve(df_spy.r,     KS)
mhat_spy_abs = mhat_curve(df_spy.abs_r, KS)
rv_spy = collect(skipmissing(df_spy.rv21)); rv_spy = rv_spy[isfinite.(rv_spy)]
mhat_spy_rv  = mhat_curve(rv_spy, KS)

# ---------- 5. Tabela CSV -----------------------------------------------------

bench = DataFrame(k = KS,
    SPY_r = mhat_spy_r,           SPY_absr = mhat_spy_abs,         SPY_RV21 = mhat_spy_rv,
    GARCH_r_q50  = band_garch.r.q50,   GARCH_r_q95  = band_garch.r.q95,
    GARCH_abs_q50= band_garch.abs.q50, GARCH_abs_q95= band_garch.abs.q95,
    GARCH_rv_q50 = band_garch.rv.q50,  GARCH_rv_q95 = band_garch.rv.q95,
    MS_r_q50     = band_ms.r.q50,      MS_r_q95     = band_ms.r.q95,
    MS_abs_q50   = band_ms.abs.q50,    MS_abs_q95   = band_ms.abs.q95,
    MS_rv_q50    = band_ms.rv.q50,     MS_rv_q95    = band_ms.rv.q95,
)
CSV.write(joinpath(OUT_DIR, "_tab_benchmarks.csv"), bench)
println("\nTabela benchmarks (SPY vs simulados):")
show(bench, allcols=true)
println()

# ---------- 6. Figura sobreposta ----------------------------------------------

gr()
default(fontfamily="Helvetica", titlefontsize=14, guidefontsize=11, tickfontsize=10,
        legendfontsize=9)

# 3 painéis lado a lado: r, |r|, RV21
function panel(title_, mhat_emp, b_garch, b_ms)
    p = plot(KS, mhat_emp, marker=:circle, lw=2.6, color=:black, label="SPY (empírico)",
             xscale=:log10, xlabel="gap k (dias)", ylabel="m̂(k)",
             title=title_, legend=:topright, ylim=(0, max(0.05, 1.05*maximum(mhat_emp))))
    # GARCH band
    plot!(p, KS, b_garch.q50, lw=2, color=RGB(0.20, 0.45, 0.75), label="GARCH(1,1) mediana")
    plot!(p, KS, b_garch.q05, fillrange=b_garch.q95, fillalpha=0.18,
          color=RGB(0.20, 0.45, 0.75), label="GARCH banda 5–95%", lw=0)
    # MS band
    plot!(p, KS, b_ms.q50, lw=2, color=RGB(0.78, 0.18, 0.15), label="MS-GARCH mediana", ls=:solid)
    plot!(p, KS, b_ms.q05, fillrange=b_ms.q95, fillalpha=0.18,
          color=RGB(0.78, 0.18, 0.15), label="MS-GARCH banda 5–95%", lw=0)
    return p
end

p_r   = panel("Retornos r",     mhat_spy_r,   band_garch.r,   band_ms.r)
p_abs = panel("|r|",            mhat_spy_abs, band_garch.abs, band_ms.abs)
p_rv  = panel("RV21",           mhat_spy_rv,  band_garch.rv,  band_ms.rv)

fig = plot(p_r, p_abs, p_rv, layout=(1,3), size=(1700, 600))
savefig(fig, joinpath(OUT_DIR, "fig_mhat_spy_benchmarks.png"))
@info "fig_mhat_spy_benchmarks.png salvo"

println("\nResumo dos parâmetros e do exercício:")
@printf("  GARCH(1,1) QMLE: ω=%.3e, α=%.3f, β=%.3f, π=α+β=%.4f\n", ω_h, α_h, β_h, α_h+β_h)
@printf("  MS-GARCH: ω₁=%.3e, κ=4.0, α=%.3f, β=%.3f, p_cc=0.995, p_aa=0.97\n", ω_h, α_h, β_h)
@printf("  B=%d trajetórias × T=%d obs (igual ao SPY)\n", B, T_sim)
