include("plot-configuration.jl")
#include("layout.jl")
include("layout-3d-ba.jl")

include("plot-utils-svm-sto.jl")
include("demo_svm_sto.jl")
include("plots-svm.jl")
include("plots-svm-smooth.jl")
include("PLM-plots-svm.jl")

include("PLM-plots-ba.jl")
include("demo_ba_sto.jl")
include("demo_ba_sto_smooth.jl")

## local work for Plot ##

#ijcnn1
include("ijcnn1-load.jl")
include("svm-plot-ijcnn1.jl")

#mnist
include("mnist-load.jl")
include("svm-plot-mnist.jl")
include("svm-mnist-greymaps-tables.jl")

#Bundle Adjustment
include("ba-load.jl")
include("ba-3d_scatter.jl")
include("ba-plots.jl")
include("ba-tables.jl")

# Plots for Objective historic, MSE and accuracy #

Random.seed!(seed)

# ---------------- Hyperbolic SVM Models ---------------- #

#include("plot-hub-svm.jl")

# ---------------- Bundle Adjustment ---------------- #

include("plot-hub-bam.jl")