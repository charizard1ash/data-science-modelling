### load libraries ###
using CSV, DataFrames, Dates, CategoricalArrays
df = DataFrames
using ZipFile
using Distributions, StatsBase, Statistics
using Colors, ColorBrewer, Plots, StatsPlots, PlotThemes
plt = Plots
using MLBase, MLJScientificTypes, MLJ, MLJTuning
mlj = MLJ


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
## scientific types
schema(hl_train)

## data analysis and transformations
# gender
df.by(hl_train, :Gender,
    count = :id => x -> length(x),
    percent = :id => x -> length(x) / df.nrow(hl_train)) # 54% male
x_1 = df.by(hl_train, :Gender, percent = :id => x -> length(x) / df.nrow(hl_train))
@df x_1 bar(:Gender, :percent;
    title = "gender distribution", xlabel = "gender", ylabel = "percent",
    color = "black", fill = (0, 0.8, :orange), legend = :none)
x_1 = nothing

# age
summarystats(hl_train[:, :Age]) # min 20, 1st quart. 25, median 36, 3rd quart. 49, max 85, mean 39
@df hl_train density(:Age;
    title = "age distribution", xlabel = "age", ylabel = "density",
    color = :black, fill = (0, 0.8, :lightgreen), legend = :none)

# driving licence
df.by(hl_train, :Driving_License,
    count = :id => x -> length(x),
    percent = :id => x -> length(x) / df.nrow(hl_train)) # ~100% licenced

# previously insured
df.by(hl_train, :Previously_Insured,
    count = :id => x -> length(x),
    percent = :id => x -> length(x) / df.nrow(hl_train)) # 46% yes

# vehicle age
df.by(hl_train, :Vehicle_Age,
    count = :id => x -> length(x),
    percent = :id => x -> length(x) / df.nrow(hl_train)) # 43% < 1 year, 53% 1-2 years

# vehicle damage
df.by(hl_train, :Vehicle_Damage,
    count = :id => x -> length(x),
    percent = :id => x -> length(x) / df.nrow(hl_train)) # 50 damaged

# region code
hl_train[!, :Region_Code_2] = ifelse.(in.(hl_train[:Region_Code], (["28","8","46","41","15","30","29","50","3"],)) .== true, hl_train[:Region_Code], "other")
df.by(hl_train, :Region_Code_2,
    count = :id => x -> length(x),
    percent = :id => x -> length(x) / df.nrow(hl_train)) # 27% 28, 38% other

# policy sales channel
hl_train[!, :Policy_Sales_Channel_2] = ifelse.(in.(hl_train[:Policy_Sales_Channel], (["152","26","124","160","156","122","157","154","151"],)) .== true, hl_train[:Policy_Sales_Channel], "other")
df.by(hl_train, :Policy_Sales_Channel_2,
    count = :id => x -> length(x),
    percent = :id => x -> length(x) / df.nrow(hl_train)) # 35% 152, 9% other

# vintage
summarystats(hl_train[:, :Vintage]) # min 10, 1st quart. 82, median 154, 3rd quart. 227, max 299, mean 154
@df hl_train density(:Vintage,
    title = "vintage distribution", xlabel = "vintage", ylabel = "density",
    color = "black", fill = (0, 0.8, :magenta), legend = :none) # almost uniform distribution

# annual premium
summarystats(hl_train[:, :Annual_Premium]) # min 2630, 1st quart. 24405, median 31669, 3rd quart. 39400, max 540165, mean 30564
@df hl_train density(:Annual_Premium,
    title = "annual premium distribution", xlabel = "annual premium", ylabel = "density",
    color = "black", fill = (0, 0.8, :lightblue), legend = :none)

# response
df.by(hl_train, :Response,
    count = :id => x -> length(x),
    percent = :id => x -> length(x) / df.nrow(hl_train)) # 12% responded

## data types
coerce!(hl_train,
    :Gender => mlj.Multiclass,
    :Age => mlj.Continuous,
    :Driving_License => mlj.Multiclass,
    :Region_Code_2 => mlj.Multiclass,
    :Vehicle_Age => mlj.Continuous,
    :Vehicle_Damage => mlj.Multiclass,
    :Policy_Sales_Channel_2 => mlj.Multiclass,
    :Previously_Insured => mlj.OrderedFactor,
    :Response => mlj.OrderedFactor)
df.levels!(hl_train[:Vehicle_Damage], ["Yes", "No"])
hl_train[:Age_Bin] = cut(hl_train[:Age], [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, Inf])
hl_train[:Vintage_Bin] = cut(hl_train[:Vintage], [0, 50, 100, 150, 200, 250, Inf])


### model development ###
## train/test split