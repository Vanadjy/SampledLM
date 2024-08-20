using RegularizedProblems, RegularizedOptimization, Test

mnist, mnist_nls, mnist_sol = RegularizedProblems.svm_train_model()
options = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-4, ϵr = 1e-4, verbose = 1, σmin = 1e-5, maxIter = 3000, maxTime = 3600.0;)
subsolver_options = RegularizedOptimization.ROSolverOptions(maxIter = 100)

## solver parameters ##
λ = 1e-1
x0 = ones(mnist.meta.nvar)
h = RootNormLhalf(λ)
l_bound = mnist.meta.lvar
u_bound = mnist.meta.uvar

@info "using R2 to solve with" h
reset!(mnist)

## LSR1 approach ##
mnist_lsr1 = LSR1Model(mnist)
R2_stats = RegularizedOptimization.R2(mnist_lsr1, h, options, x0 = x0)

## RegularizedNLSModel approach ##
reg_prob = RegularizedNLSModel(mnist_nls, h)
reg_stats = GenericExecutionStats(reg_prob.model)
reg_solver = RegularizedOptimization.R2Solver(x0, options, l_bound, u_bound; ψ = shifted(h, x0))
cb = (nlp, solver, stats) -> begin
                                solver.Fobj_hist[stats.iter+1] = stats.solver_specific[:smooth_obj] + stats.solver_specific[:nonsmooth_obj]
                                solver.Hobj_hist[stats.iter+1] = stats.solver_specific[:xi]
                                solver.Complex_hist[stats.iter+1] += 1
                             end
RegularizedOptimization.solve!(reg_solver, reg_prob, reg_stats; callback = cb)

@test reg_stats.iter == R2_stats.iter