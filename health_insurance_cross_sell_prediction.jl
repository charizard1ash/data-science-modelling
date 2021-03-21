### load libraries ###
using CSV, DataFrames, Dates, CategoricalArrays
df = DataFrames
ca = CategoricalArrays
using ZipFile
using Distributions, StatsBase, Statistics
sb = StatsBase
st = Statistics
using Colors, ColorBrewer, Plots, StatsPlots, PlotThemes
plt = Plots
using MLBase, MLJScientificTypes, MLJ, MLJTuning
mlj = MLJ
@load DecisionTreeClassifier pkg = "DecisionTree"
@load RandomForestClassifier pkg = "DecisionTree"
@load LogisticClassifier pkg = "MLJLinearModels"
@load XGBoostClassifier pkg = "XGBoost"
using ShapML
shap = ShapML
using Base.Threads


### set variables ###
data_location = "..."


### set functions ###
## train/test split
function sample_split(dt::DataFrames.DataFrame, percent::Float64=0.7)
    n_train = sb.sample(collect(1:1:df.nrow(dt)), Int64(floor(df.nrow(dt) * percent)); replace = false)
    n_test_df = df.DataFrame(x1=collect(1:1:df.nrow(dt)))
    n_test_df[!, :x2] = ifelse.(in.(n_test_df[:x1], (n_train, )) .== true, 0, 1)
    n_test = n_test_df[n_test_df[:x2] .== 1, :x1]
    # n_test = Int64[]
    # for i in collect(1:1:df.nrow(dt))
    #     if in(i, n_train) == false
    #         push!(n_test, i)
    #     end
    # end
    return n_train, n_test
end


### import data ###
## health train
hl_train = CSV.read(string(data_location, "train.csv"), df.DataFrame; delim=",", header=1)
hl_train[!, :Region_Code] = string.(convert.(Int64, hl_train[:, :Region_Code]))
hl_train[!, :Policy_Sales_Channel] = string.(convert.(Int64, hl_train[:, :Policy_Sales_Channel]))

## health test
hl_test = CSV.read(string(data_location, "test.csv"), df.DataFrame; delim=",", header=1)
hl_test[!, :Region_Code] = string.(convert.(Int64, hl_test[:, :Region_Code]))
hl_test[!, :Policy_Sales_Channel] = string.(convert.(Int64, hl_test[:, :Policy_Sales_Channel]))

## health submission
hl_submission = CSV.read(string(data_location, "sample_submission.csv"), df.DataFrame; delim=",", header=1)


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
@df x_1 plt.bar(:Gender, :percent;
    title = "gender distribution", xlabel = "gender", ylabel = "percent",
    color = "black", fill = (0, 0.8, :orange), legend = :none)
x_1 = nothing

# age
sb.summarystats(hl_train[:, :Age]) # min 20, 1st quart. 25, median 36, 3rd quart. 49, max 85, mean 39
@df hl_train plt.density(:Age;
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
sb.summarystats(hl_train[:, :Vintage]) # min 10, 1st quart. 82, median 154, 3rd quart. 227, max 299, mean 154
@df hl_train plt.density(:Vintage,
    title = "vintage distribution", xlabel = "vintage", ylabel = "density",
    color = "black", fill = (0, 0.8, :magenta), legend = :none) # almost uniform distribution

# annual premium
sb.summarystats(hl_train[:, :Annual_Premium]) # min 2630, 1st quart. 24405, median 31669, 3rd quart. 39400, max 540165, mean 30564
@df hl_train plt.density(:Annual_Premium,
    title = "annual premium distribution", xlabel = "annual premium", ylabel = "density",
    color = "black", fill = (0, 0.8, :lightblue), legend = :none)

# response
df.by(hl_train, :Response,
    count = :id => x -> length(x),
    percent = :id => x -> length(x) / df.nrow(hl_train)) # 12% responded

## data types
# train
coerce!(hl_train,
    :Gender => mlj.Multiclass,
    :Age => mlj.Continuous,
    :Driving_License => mlj.OrderedFactor,
    :Region_Code_2 => mlj.Multiclass,
    :Vehicle_Age => mlj.Multiclass,
    :Vehicle_Damage => mlj.Multiclass,
    :Policy_Sales_Channel_2 => mlj.Multiclass,
    :Previously_Insured => mlj.OrderedFactor,
    :Response => mlj.OrderedFactor)
ca.levels!(hl_train[:Gender], ["Male","Female"])
ca.levels!(hl_train[:Region_Code_2], ["28","8","46","41","15","30","29","50","3","other"])
ca.levels!(hl_train[:Vehicle_Age], ["< 1 Year","1-2 Year","> 2 Years"])
ca.levels!(hl_train[:Vehicle_Damage], ["Yes","No"])
ca.levels!(hl_train[:Policy_Sales_Channel_2], ["152","26","124","160","156","122","157","154","151","other"])
hl_train[:Age_Bin] = ca.cut(hl_train[:Age], [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, Inf])
hl_train[:Vintage_Bin] = ca.cut(hl_train[:Vintage], [0, 50, 100, 150, 200, 250, Inf])
hl_train[:Annual_Premium_Bin] = ca.cut(hl_train[:Annual_Premium], [0, 20000, 25000, 30000, 35000, 40000, 45000, 50000, Inf])

# test
hl_test[!, :Region_Code_2] = ifelse.(in.(hl_test[:Region_Code], (["28","8","46","41","15","30","29","50","3"],)) .== true, hl_test[:Region_Code], "other")
hl_test[!, :Policy_Sales_Channel_2] = ifelse.(in.(hl_test[:Policy_Sales_Channel], (["152","26","124","160","156","122","157","154","151"],)) .== true, hl_test[:Policy_Sales_Channel], "other")
coerce!(hl_test,
    :Gender => mlj.Multiclass,
    :Age => mlj.Continuous,
    :Driving_License => mlj.OrderedFactor,
    :Region_Code_2 => mlj.Multiclass,
    :Vehicle_Age => mlj.Multiclass,
    :Vehicle_Damage => mlj.Multiclass,
    :Policy_Sales_Channel_2 => mlj.Multiclass,
    :Previously_Insured => mlj.OrderedFactor,
    :Response => mlj.OrderedFactor)
ca.levels!(hl_test[:Gender], ["Male","Female"])
ca.levels!(hl_test[:Region_Code_2], ["28","8","46","41","15","30","29","50","3","other"])
ca.levels!(hl_test[:Vehicle_Age], ["< 1 Year","1-2 Year","> 2 Years"])
ca.levels!(hl_test[:Vehicle_Damage], ["Yes","No"])
ca.levels!(hl_test[:Policy_Sales_Channel_2], ["152","26","124","160","156","122","157","154","151","other"])
hl_test[:Age_Bin] = ca.cut(hl_test[:Age], [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, Inf])
hl_test[:Vintage_Bin] = ca.cut(hl_test[:Vintage], [0, 50, 100, 150, 200, 250, Inf])
hl_test[:Annual_Premium_Bin] = ca.cut(hl_test[:Annual_Premium], [0, 20000, 25000, 30000, 35000, 40000, 45000, 50000, Inf])


### model development ###
## train/test split
# note we use relevant features
# core train dataset
hl_train_2 = df.select(hl_train, [:id, :Gender, :Age_Bin, :Driving_License, :Region_Code_2, :Previously_Insured, :Vehicle_Age, :Vehicle_Damage, :Annual_Premium_Bin, :Policy_Sales_Channel_2, :Vintage_Bin, :Response])

# core test dataset
hl_test_2 = df.select(hl_train, [:id, :Gender, :Age_Bin, :Driving_License, :Region_Code_2, :Previously_Insured, :Vehicle_Age, :Vehicle_Damage, :Annual_Premium_Bin, :Policy_Sales_Channel_2, :Vintage_Bin, :Response])

# split train
n_train, n_test = mlj.partition(1:df.nrow(hl_train_2), 0.8; shuffle=true)
hl_model_train = hl_train_2[n_train, :]
hl_model_test = hl_train_2[n_test, :]

## core training dataset
y, X = mlj.unpack(hl_model_train, ==(:Response), colname -> true)
df.select!(X, Not(:id))

# one hot encoded predictor variables
Xt = mlj.transform(mlj.fit!(mlj.machine(OneHotEncoder(drop_last = true), X)))

# search for models
# for i in collect(1:1:length(mlj.models(matching(Xt, y)))) println(mlj.models(matching(Xt, y))[i]) end

# check models
# mlj.models(matching(X, y))
# mlj.models(matching(Xt, y))

## glm
mlj.evaluate(LogisticClassifier(), Xt, y, resampling = Holdout(fraction_train = 0.7, shuffle=true), measures = [auc, cross_entropy])
mlj.evaluate(LogisticClassifier(), Xt, y, resampling = CV(nfolds = 5, shuffle = true), measures = [auc, cross_entropy])
hl_glm = mlj.machine(LogisticClassifier(), Xt, y)
mlj.fit!(hl_glm; force = true)
mlj.fitted_params(hl_glm)
mlj.report(hl_glm)

## decision tree
# mlj.info(DecisionTreeClassifier)
r1 = mlj.range(DecisionTreeClassifier(), :max_depth; values = [10, 20, 30])
r2 = mlj.range(DecisionTreeClassifier(), :min_samples_leaf; values = [25, 50])
r3 = mlj.range(DecisionTreeClassifier(), :min_samples_split; values = [25, 50])
r4 = mlj.range(DecisionTreeClassifier(), :min_purity_increase; values = [0.0, 0.01, 0.05])
dt_tune = TunedModel(;model = DecisionTreeClassifier(),
    tuning = Grid(),
    resampling = Holdout(fraction_train = 0.7, shuffle = true),
    measure = [auc, cross_entropy],
    repeats = 1,
    range = [r1, r2, r3, r4])
hl_dt = mlj.machine(dt_tune, Xt, y)
println(now())
mlj.fit!(hl_dt; force = true)
(r1, r2, r3, r4) = (nothing, nothing, nothing, nothing)
println(now())
mlj.save(string(data_location, "hl_dt.jlso"), hl_dt)
mlj.fitted_params(hl_dt)
mlj.report(hl_dt).best_model
mlj.report(hl_dt).best_result
mlj.report(hl_dt).best_report
mlj.report(hl_dt).history
mlj.report(hl_dt).plotting

## random RandomForestClassifier
# mlj.info(RandomForestClassifier)
r1 = mlj.range(RandomForestClassifier(), :max_depth; values = [10, 20])
r2 = mlj.range(RandomForestClassifier(), :min_samples_leaf; values = [25, 50])
r3 = mlj.range(RandomForestClassifier(), :min_samples_split; values = [25, 50])
r4 = mlj.range(RandomForestClassifier(), :min_purity_increase; values = [0.00, 0.05])
r5 = mlj.range(RandomForestClassifier(), :n_trees; values = [50, 100, 150])
rf_tune = TunedModel(; model = RandomForestClassifier(),
    tuning = Grid(),
    resampling = Holdout(fraction_train = 0.7, shuffle = true),
    measure = [auc, cross_entropy],
    repeats = 1,
    range = [r1, r2, r3, r4, r5])
hl_rf = mlj.machine(rf_tune, Xt, y)
println(now())
mlj.fit!(hl_rf; force = true)
(r1, r2, r3, r4, r5) = (nothing, nothing, nothing, nothing, nothing)
println(now())
mlj.save(string(data_location, "hl_rf.jlso"), hl_rf)
mlj.fitted_params(hl_rf)
mlj.report(hl_rf).best_model
mlj.report(hl_rf).best_result
mlj.report(hl_rf).best_report
mlj.report(hl_rf).history
mlj.report(hl_rf).plotting
