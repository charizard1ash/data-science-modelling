### Load libraries ###
using CSV, DataFrames, Dates, RDatasets;
df = DataFrames;
using Colors, ColorBrewer;
using Plots, StatsPlots;
plt = Plots;
using CategoricalArrays, StatsBase, Statistics, Distributions, HypothesisTests;
using GLM, DecisionTree, LIBSVM, XGBoost, Flux;


### Set variables ###
jl_func_location = "C:/Users/nss6/Documents/projects/charizard1ash/julia-functions/";


### Import functions ###
include(string(jl_func_location, "roc.jl"));


### Import data ###
iris = RDatasets.dataset("datasets", "iris");
iris = df.DataFrame(iris);
println(iris[1:5, :]);


### Conduct analysis ###
## aggregate.
@time print(df.aggregate(iris, :Species, mean))

## test by function.
@time print(df.by(iris, :Species, [:PetalLength, :PetalWidth, :SepalLength, :SepalWidth] =>
    x -> (count=length(x[:PetalLength]), mean_petal_length=mean(x[:PetalLength]), mean_petal_width=mean(x[:PetalWidth]))))

@time print(df.by(iris, :Species, [:PetalLength, :PetalWidth] =>
    x -> (mean_petal_length=mean(x[:PetalLength]), mean_petal_width=mean(x[:PetalWidth]))))

## test plotting.
@time @df iris andrewsplot(:Species, cols(1:4),
    line=(fill=[:blue :red :green]), legend=:topleft)

@time @df iris plt.scatter(:SepalLength, :SepalWidth,
    group=:Species,
    title="sepal length vs sepal width", xlabel="sepal length", ylabel="sepal width",
    legend=:topleft)

@time @df iris plt.scatter(:SepalLength, :SepalWidth, :PetalLength;
    group=:Species,
    title="3d scatter plot", xlabel="sepal length", ylabel="sepal width", zlabel="petal length",
    legend=:topright)


### Build various machine learning models ###
## create binary flags for each species
iris[!, :setosa_flag] = ifelse.(iris[!, :Species] .== "setosa", 1, 0)
iris[!, :versicolor_flag] = ifelse.(iris[!, :Species] .== "versicolor", 1, 0)
iris[!, :virginica_flag] = ifelse.(iris[!, :Species] .== "virginica", 1, 0)

## split into training and test
n_index = StatsBase.sample(1:df.size(iris)[1], Int(floor(df.size(iris)[1]*0.70)); replace=false)
df_train = iris[n_index, :]
df_test = iris[setdiff(1:df.size(iris)[1], n_index), :]

## create default roc and auc dataframes.
model_roc = df.DataFrame(model_name=String[], prob_threshold=Float64[], tnr=Float64[], fpr=Float64[], fnr=Float64[], tpr=Float64[], precision=Float64[], recall=Float64[], f1_score=Float64[])
model_auc = df.DataFrame(model_name=String[], auc=Float64[])

## glm
@time glm_iris = GLM.glm(@formula(versicolor_flag ~ SepalLength + SepalWidth + PetalLength + PetalWidth), df_train, Binomial());
model_prob = GLM.predict(glm_iris, df_test)
model_roc_temp, model_auc_temp = roc(model_prob, df_test[:, :versicolor_flag]; prob_thresh=collect(0:0.01:1));
model_roc_temp[!, :model_name] .= "glm 01"
model_auc_temp = df.DataFrame(model_name="glm 01", auc=model_auc_temp)
model_roc = vcat(model_roc, model_roc_temp)
model_auc = vcat(model_auc, model_auc_temp)
model_roc_temp, model_auc_temp = nothing, nothing