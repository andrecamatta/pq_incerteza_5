# pq_incerteza_5 — Além do risco, Parte 5

Código de apoio ao artigo "Além do risco — Parte 5: o mercado se mistura?", da série
*Pílulas de Quant* (pilulasdequant.com.br).

O artigo testa, em ETFs (SPY, TLT, EFA, EEM, EWZ) com séries diárias da Tiingo até
2025-12-31, implicações empíricas das condições de mixing apresentadas teoricamente na
[Parte 4 da série](https://pilulasdequant.com.br/p/alem-do-risco-parte-4-componentes).
Camadas de diagnóstico:

- **Linear**: ACF, PACF, Ljung-Box, soma parcial Σ|γ̂(h)|.
- **Risco**: ACF de |r| e r², ARCH-LM, meia-vida de log(RV<sub>21</sub>).
- **Mais próxima de mixing**: quase-coeficiente empírico m̂(k) por discretização em
  quartis e bandas iid via permutação.

## Estrutura

```
code/
  Project.toml          # ambiente Julia
  download_tiingo.jl    # baixa preços ajustados de SPY, TLT, EFA, EEM, EWZ
  analysis.jl           # módulo com diagnósticos (acf_at, ljung_box, arch_lm, mhat, ...)
  run.jl                # roda diagnósticos e gera CSVs de tabelas + figuras
  figure_main.jl        # figura central com bandas iid próprias para cada série
data/
  *.csv                 # séries baixadas (não-versionado por padrão)
```

## Como reproduzir

1. Coloque a chave da Tiingo em `~/.claude/commands/tiingo.key` (uma linha de texto).
2. Instale o ambiente Julia:
   ```bash
   cd code
   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   ```
3. Baixe os dados:
   ```bash
   julia --project=. download_tiingo.jl
   ```
4. Rode os diagnósticos:
   ```bash
   julia --project=. run.jl
   ```
5. Gere a figura central:
   ```bash
   julia --project=. figure_main.jl
   ```

Saídas em `../artigos/`: tabelas em CSV (`_tab_*.csv`) e figuras (`fig_*.png`).
