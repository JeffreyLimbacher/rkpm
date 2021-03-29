using LinearAlgebra
using Plots
using VoronoiDelaunay

function get_domain(x_pts)
    np = length(x_pts)
    x = zeros(2, np, np)
    x[1, :, :] .= x_pts
    x[2, :, :] .= x_pts'
    reshape(x, 2, np * np)
end

function get_Hx_from_diff_mat(x_x_I, order)
    function get_Hx!(x_x_I, order, hx)
        k = 1
        for i=0:order
            for j=0:(order-i)
                hx[k] = prod(x_x_I[1]^i * x_x_I[2]^j)
                k += 1
            end
        end
    end
    Hx_size = Int((order+1)*(order+2)/2)
    n_eval = size(x_x_I, 2)
    n_pts = size(x_x_I, 3)
    Hx = zeros(Hx_size, n_eval, n_pts)
    Threads.@threads for i=1:n_pts
        for j=1:n_eval
            @views get_Hx!(x_x_I[:,i,j], order, Hx[:, i, j])
        end
    end
    Hx
end

function get_diff_matrix(x_I, x)
    x_I_r = reshape(X_I,2,1,size(X_I,2))
    x_I_l = reshape(X_I,2,size(X_I,2),1)
    x_I_l .- x_I_r
end

function get_coefficients(x_x_I, Hx_x_I, phi_a, order)
    n_eval = size(x_x_I, 2)
    Hx_size = size(Hx_x_I, 1)
    B = zeros(Hx_size, n_eval)
    H0 = zeros(Hx_size)
    H0[1] = 1
    Threads.@threads for i=1:n_eval
        # eq. 5.9
        H_np = Hx_x_I[:, i, :]
        Mx = H_np * (H_np' .* phi_a[i, :])
        B[:,i] = Mx \ H0
    end
    B
end

function calc_psi(B, Hx_x_I, phi_a)
    n_eval = size(phi_a, 1)
    n_pts = size(phi_a, 2)
    psi = zeros(n_pts, n_pts)
    Threads.@threads for i=1:n_eval
        for j=1:n_pts
            # eq (5.10)
            psi[j,i] = B[:,j]' * Hx_x_I[:, j, i] .* phi_a[j,i]
        end
    end
    psi
end


function rkpm_shape_funcs(x::Matrix{Float64}, order::Int, a::Float64)
    @time x_x_I = get_diff_matrix(x, x)
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
    dropdims(bspline.(sqrt.(sum(x.^2,dims=1))/a)/a;dims=1)
end

step_size = .1
domain = 0:step_size:5
domain_sample = domain[1:1:end]

# our test function
f = (x,y) -> cos(x * pi) + sin(y * pi)
X_I = get_domain(domain_sample)
n_pts = size(X_I,1)
Y_I = f.(X_I[1,:], X_I[2,:])
X = get_domain(domain)
Y = f.(X[:,1], X[:,2])
n_eval = size(X,1)

#n_pts is number of interpolation points
#n_eval is true function and where evaluate interpolation
# support size
a = 0.3 .* (X_I[end]-X_I[1])

psi = rkpm_shape_funcs(X_I, 3, a)
u_I = psi * Y_I
surface(X[:,1], X[:,2], u_I, legend=false)
