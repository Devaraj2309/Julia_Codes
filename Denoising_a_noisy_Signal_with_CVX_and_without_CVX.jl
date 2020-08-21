using DelimitedFiles
using SparseArrays
using LinearAlgebra
using Plots
using Convex, SCS

y=readdlm("data.txt", '\n')
N =length(y)
ey = Matrix{Float64}(I,N,N)
D = Matrix(spdiagm(N-2, N, 2 => ones(N-2),1 => -2*ones(N-2), 0 => ones(N-2)))
lam = 100

x=(ey+lam*D'*D)\y
Ty=(D*y)'*(D*y)
Tx=(D*x)'*(D*x)

plot(y,label ="Noisy signal")

p2 = plot(x,label = "Denoised signal")

x1 = Variable(N)
problem = minimize(sumsquares(y - x1)+lam*sumsquares(D*x1))
# Solve the problem by calling solve!
solve!(problem, SCS.Optimizer)
# Check the status of the problem
problem.status 
# Get the optimal value
problem.optval

x1.value
plot(x1.value,label = "Denoised using cvx")


