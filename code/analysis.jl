"""
Diagnósticos de mixing/ergodicidade em séries financeiras.

Pacote-âncora do artigo "Além do risco — Parte 5".
Para cada ETF (SPY, TLT, EFA, EEM, EWZ) computa:
  • ACF/PACF de r, |r|, r² em lags 1, 5, 21, 63
  • Ljung–Box em retornos e em |r|
  • ARCH-LM (Engle 1982) de ordem 5 e 22
  • Soma parcial Σ_{h=1}^{H} |γ̂(h)| para H = {21, 63, 252}, normalizada por γ̂(0)
  • Half-life de choques de log(RV21) via AR(1)
  • Quase-coeficiente de mixing empírico m̂(k) por discretização em quartis,
    para k em {1,5,10,21,42,63,126,252} e blocos de tamanho m=1.
"""
module Analysis

using DataFrames, CSV, Statistics, StatsBase, LinearAlgebra
using Distributions, Printf, Dates, Random

export load_etf, acf_at, ljung_box, arch_lm, autocov_partial_sum,
       halflife_logrv, mhat, mhat_iid_band

# ---------- Loading & transforms ----------------------------------------------

"Lê CSV (date, adjClose) e devolve DataFrame com r, abs_r, r2 e RV21."
function load_etf(path::String)
    df = CSV.read(path, DataFrame)
    sort!(df, :date)
    logp = log.(df.adjClose)
    r = diff(logp)
    df = df[2:end, :]
    df.r = collect(r)               # Vector{Float64}, sem missing
    df.abs_r = abs.(df.r)
    df.r2 = df.r .^ 2
    # RV21: soma móvel de r² em janela 21d (não anualizada)
    n = nrow(df)
    rv = fill(NaN, n)
    @inbounds for i in 21:n
        rv[i] = sum(view(df.r2, (i-20):i))
    end
    df.rv21 = rv
    return df
end

# ---------- Estatísticas serial -----------------------------------------------

"ACF amostral em lag h via correlação direta (corrigida pela média)."
function acf_at(x::AbstractVector{<:Real}, h::Int)
    n = length(x)
    μ = mean(x)
    γ0 = sum((x .- μ).^2) / n
    γh = sum((x[(h+1):n] .- μ) .* (x[1:(n-h)] .- μ)) / n
    return γh / γ0
end

"Soma parcial Σ_{h=1}^{H} |γ̂(h)| / γ̂(0)."
function autocov_partial_sum(x::AbstractVector{<:Real}, H::Int)
    n = length(x)
    μ = mean(x)
    γ0 = sum((x .- μ).^2) / n
    s = 0.0
    @inbounds for h in 1:H
        γh = sum((x[(h+1):n] .- μ) .* (x[1:(n-h)] .- μ)) / n
        s += abs(γh)
    end
    return s / γ0
end

"Estatística de Ljung-Box até lag m, devolve (Q, df=m, p-valor)."
function ljung_box(x::AbstractVector{<:Real}, m::Int)
    n = length(x)
    Q = 0.0
    @inbounds for h in 1:m
        ρh = acf_at(x, h)
        Q += ρh^2 / (n - h)
    end
    Q *= n * (n + 2)
    p = ccdf(Chisq(m), Q)
    return (Q=Q, df=m, p=p)
end

"""
ARCH-LM test (Engle 1982).
Regride r²_t em constante + r²_{t-1}, ..., r²_{t-q}; estatística LM = T·R²,
distribuída χ²(q) sob H0 de homoscedasticidade.
"""
function arch_lm(r::AbstractVector{<:Real}, q::Int)
    r2 = r .^ 2
    T = length(r2) - q
    y = r2[(q+1):end]
    X = [ones(T) [r2[(q+1-j):(end-j)] for j in 1:q]...]
    X = hcat(ones(T), reduce(hcat, [r2[(q+1-j):(end-j)] for j in 1:q]))
    β = X \ y
    yhat = X * β
    sse = sum((y .- yhat).^2)
    sst = sum((y .- mean(y)).^2)
    R2 = 1 - sse / sst
    LM = T * R2
    p = ccdf(Chisq(q), LM)
    return (LM=LM, df=q, p=p, R2=R2)
end

"""
Half-life de choques em log(RV21) via AR(1):
log(RV_t) = c + φ·log(RV_{t-1}) + u_t.
Half-life = log(0.5)/log(φ) (apenas se 0<φ<1).
"""
function halflife_logrv(rv::AbstractVector{<:Real})
    rv = collect(skipmissing(rv))
    rv = rv[isfinite.(rv) .& (rv .> 0)]
    y = log.(rv)
    yt = y[2:end]; ytm1 = y[1:end-1]
    X = hcat(ones(length(yt)), ytm1)
    β = X \ yt
    φ = β[2]
    hl = (0 < φ < 1) ? log(0.5)/log(φ) : NaN
    return (φ=φ, halflife_days=hl)
end

# ---------- Quase-coeficiente de mixing empírico ------------------------------

"""
m̂(k; Q) = max_{i,j} |P̂(X_t ∈ Q_i, X_{t+k} ∈ Q_j) − P̂(X_t ∈ Q_i)·P̂(X_{t+k} ∈ Q_j)|

Discretização: divide x em Q quartis (Q=4). Computa tabela 2D de
frequências conjuntas e marginais empíricas, devolve o desvio máximo absoluto.

Nota: aproximação simples e não corrigida para vício de tamanho amostral.
Pode ser comparada contra um benchmark de bootstrap iid (não implementado aqui).
"""
function mhat(x::AbstractVector{<:Real}, k::Int; Q::Int=4)
    n = length(x) - k
    n > 100 || return NaN
    qs = quantile(x, range(0, 1; length=Q+1)[2:end-1])
    bin(v) = begin
        b = 1
        @inbounds for q in qs
            v > q ? (b += 1) : break
        end
        b
    end
    counts = zeros(Q, Q)
    @inbounds for t in 1:n
        i = bin(x[t]); j = bin(x[t+k])
        counts[i, j] += 1
    end
    Pjoint = counts / n
    Pi = sum(Pjoint, dims=2)
    Pj = sum(Pjoint, dims=1)
    Pind = Pi * Pj
    return maximum(abs.(Pjoint .- Pind))
end

"""
Bootstrap iid (permutação) para uma banda de m̂ sob hipótese de independência.
Devolve quantis 5%, 50%, 95% após `B` permutações.
"""
function mhat_iid_band(x::AbstractVector{<:Real}, k::Int; Q::Int=4, B::Int=200, rng=Random.GLOBAL_RNG)
    n = length(x)
    m = Vector{Float64}(undef, B)
    @inbounds for b in 1:B
        xperm = x[Random.shuffle(rng, 1:n)]
        m[b] = mhat(xperm, k; Q=Q)
    end
    return (q05 = quantile(m, 0.05), q50 = median(m), q95 = quantile(m, 0.95))
end

end # module
