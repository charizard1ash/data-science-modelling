### load libraries ###
using CSV, DataFrames, Dates, CategoricalArrays
df = DataFrames
using ZipFile
using Distributions, StatsBase, Statistics
using Colors, ColorBrewer, Plots, StatsPlots, PlotThemes
using MLBase, MLJScientificTypes, MLJ, MLJTuning


### set variables ###
data_location = "C:/Users/nss6/Documents/Data/Kaggle/health-insurance-cross-sell-prediction/"


### import data ###
## health train
hl_train = CSV.read(string(data_location, "train.csv"); delim=",", header=1)
hl_train[!, :Region_Code] = string.(convert.(Int64, hl_train[:, :Region_Code]))
hl_train[!, :Policy_Sales_Channel] = string.(convert.(Int64, hl_train[:, :Policy_Sales_Channel]))

## health test
hl_test = CSV.read(string(data_location, "test.csv"); delim=",", header=1)
hl_test[!, :Region_Code] = string.(convert.(Int64, hl_test[:, :Region_Code]))
hl_test[!, :Policy_Sales_Channel] = string.(convert.(Int64, hl_test[:, :Policy_Sales_Channel]))

## health submission
hl_submission = CSV.read(string(data_location, "sample_submission.csv"); delim=",", header=1)


### exploratory data analysis ###
## null counts
# train
println(df.describe(hl_train)[:, [Symbol("variable"), Symbol("nmissing")]])

# test
println(df.describe(hl_test)[:, [Symbol("variable"), Symbol("nmissing")]])

# submission
println(df.describe(hl_submission)[:, [Symbol("variable"), Symbol("nmissing")]])

## distributions
x_1 = df.by(hl_train, :Region_Code, 
    count = :id => length,
    percent = :id => x -> length(x) / df.nrow(hl_train))
df.sort!(x_1, (order(:count, rev = true), :Region_Code))
println(x_1) # keep region codes 28, 8, 46, 41, 15, 30, 29, 50, and 3. remaining can be categorised as other
x_1 = nothing

x_1 = df.by(hl_train, :Policy_Sales_Channel,
    count = :id => length,
    percent = :id => x -> length(x) / df.nrow(hl_train))
df.sort!(x_1, (order(:count, rev = true), :Policy_Sales_Channel))
println(x_1) # keep policy sales channels 152, 26, 124, 160, 156, 122, 157, 154, and 151. remaining can be categorised as other


### feature engineering ###
hl_train[!, :Region_Code_2] = ifelse.(in.(hl_train[:Region_Code], (["28","8","46","41","15","30","29","50","3"],)) .== true, hl_train[:Region_Code], "other")
println(df.by(hl_train, :Region_Code_2,
    count = :id => x -> length(x),
    percent = :id => x -> length(x) / df.nrow(hl_train)))
hl_train[!, :Policy_Sales_Channel_2] = ifelse.(in.(hl_train[:Policy_Sales_Channel], (["152","26","124","160","156","122","157","154","151"],)) .== true, hl_train[:Policy_Sales_Channel], "other")
println(df.by(hl_train, :Policy_Sales_Channel_2,
    count = :id => x -> length(x),
    percent = :id => x -> length(x) / df.nrow(hl_train)))
