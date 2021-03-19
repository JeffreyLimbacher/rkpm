using LinearAlgebra
using Plots
using VoronoiDelaunay

function get_domain(x_pts)
    np = length(x_pts)
    x = zeros(np, np, 2)
    x[:, :, 1] .= x_pts
    x[:, :, 2] .= x_pts'
    reshape(x, np * np, 2)
end

#x_x_I is the vector of differences between the points
function get_Hx(x_x_I, order)
    exps = [(i, j) for i=0:order for j=0:order if i+j<=order]
    hx = zeros(length(exps))
    for (i,exp) in enumerate(exps)
        hx[i] = prod(x_x_I.^exp)
    end
    hx
end

function get_Hx_from_diff_mat(x_x_I, order)
    Hx_size = Int((order+1)*(order+2)/2)
    n_eval = size(x_x_I, 1)
    n_pts = size(x_x_I, 2)
    Hx = zeros(Hx_size, n_eval, n_pts)
    Threads.@threads for i=1:n_pts
        for j=1:n_eval
            Hx[:, i, j] = get_Hx(x_x_I[i,j,:], order)
        end
    end
    Hx
end

function get_diff_matrix(x_I, x)
    n_pts = size(x, 1)
    n_eval = size(x_I, 1)
    x_x_I = zeros(n_eval, n_pts, 2)
    for i=1:2
        x_x_I[:,:,i] = X[:,i] .- X_I[:,i]'
    end
    x_x_I
end

function get_coefficients(x_x_I, Hx_x_I, phi_a, order)
    n_eval = size(x_x_I, 1)
    Hx_size = size(Hx_x_I, 1)
    B = zeros(Hx_size, n_eval)
    H0 = zeros(Hx_size)
    H0[1] = 1
    Threads.@threads for i=1:n_eval
        Mx = zeros(Hx_size, Hx_size)
        for j=1:n_pts
            H_np = Hx_x_I[:, i, j]
            H_np2 = H_np * H_np'
            Mx += H_np2 .* phi_a[i,j]
        end
        # eq. 5.9
        B[:,i] = Mx \ H0
    end
    B
end

function calc_psi(B, Hx_x_I, phi_a)
    n_eval = size(phi_a, 2)
    n_pts = size(phi_a, 2)
    psi = zeros(n_pts, n_pts)
    Threads.@threads for i=1:n_pts
        for j=1:n_pts
            # eq (5.10)
            psi[j,i] = B[:,j]' * Hx_x_I[:, j, i] .* phi_a[j,i]
        end
    end
    psi
end


function rkpm_shape_funcs(x::Matrix{Float64}, x_I::Matrix{Float64}, order::Int, a::Float64)
    @time x_x_I = get_diff_matrix(x, x_I)
    @time Hx_x_I = get_Hx_from_diff_mat(x_x_I, order)
    @time phi_a = basis(x_x_I, a)
    @time B = get_coefficients(x_x_I, Hx_x_I, phi_a, order)
    @time calc_psi(B, Hx_x_I, phi_a)
end

function bspline(z)
    zabs = abs(z)
    if zabs <= 0.5
        2.0/3.0 - 4z^2 + 4z^3
    elseif zabs <= 1.0
        4.0/3.0 * (1-z)^3
    else
        0
    end
end

function basis(x, a)
    bspline.(sqrt.(sum(x.^2,dims=3))/a)/a
end

step_size = .1
domain = 0:step_size:5
domain_sample = domain[1:1:end]

# our test function
f = (x,y) -> cos(x * pi) + sin(y * pi)
X_I = get_domain(domain_sample)
n_pts = size(X_I,1)
Y_I = f.(X_I[:,1], X_I[:,2])
X = get_domain(domain)
Y = f.(X[:,1], X[:,2])
n_eval = size(X,1)

#n_pts is number of interpolation points
#n_eval is true function and where evaluate interpolation
# support size
a = 0.3 .* (X_I[end]-X_I[1])

psi = rkpm_shape_funcs(X_I, X_I, 2, a)
u_I = psi * Y_I


#surface(X[:,1], X[:,2], u_h[:,1])
surface(X[:,1], X[:,2], u_I, legend=false)
#@show sum(abs.(u_h-Y))/sum(abs.(Y))
#p=scatter(X[:,1], X[:,2], zeros(length(X[:,1])))
