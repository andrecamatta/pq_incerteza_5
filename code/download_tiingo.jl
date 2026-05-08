# Download diário Tiingo dos 5 ETFs
using HTTP, JSON3, DataFrames, CSV, Dates

const TIINGO_KEY = strip(read(joinpath(homedir(), ".claude", "commands", "tiingo.key"), String))

const TICKERS = ["SPY", "TLT", "EFA", "EEM", "EWZ"]
const END_DATE = "2025-12-31"

function fetch_tiingo(ticker::String)::DataFrame
    url = "https://api.tiingo.com/tiingo/daily/$(lowercase(ticker))/prices"
    headers = ["Authorization" => "Token $TIINGO_KEY", "Content-Type" => "application/json"]
    query = ["startDate" => "1990-01-01", "endDate" => END_DATE, "format" => "json"]
    resp = HTTP.get(url, headers; query=query, readtimeout=120, retry=true)
    data = JSON3.read(resp.body)
    df = DataFrame(
        date = [Date(SubString(d.date, 1, 10)) for d in data],
        adjClose = [Float64(d.adjClose) for d in data],
    )
    sort!(df, :date)
    return df
end

mkpath("../data")
for t in TICKERS
    @info "Baixando $t ..."
    df = fetch_tiingo(t)
    CSV.write("../data/$(lowercase(t)).csv", df)
    @info "  $t: $(nrow(df)) linhas, $(df.date[1])$(df.date[end])"
end

println("OK")
