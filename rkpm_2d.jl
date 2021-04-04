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

function get_Hx_size(order, dim)
    if dim == 1
        order
    elseif dim == 2
        Int((order+1)*(order+2)/2)
    elseif dim == 3
        Int((order+1)*(order+2)*(order+3)/6)
    else
        throw("Only supports 1, 2, or 3 dimensions")
    end
end

function get_Hx_from_diff_mat(x_x_I, order)
    function get_Hx!(x_x_I, order, hx)
        k = 1
        for i=0:order
            for j=0:(order-i)
                hx[k] = x_x_I[1]^i * x_x_I[2]^j
                k += 1
            end
        end
    end
    Hx_size = Int((order+1)*(order+2)/2)
    n_eval = size(x_x_I, 2)
    n_pts = size(x_x_I, 3)
    Hx = zeros(Hx_size, n_eval, n_pts)
    Threads.@threads for j=1:n_pts
        for i=1:n_eval
            @views get_Hx!(x_x_I[:,i,j], order, Hx[:, i, j])
        end
    end
    Hx
end

function get_Hx(x_x_I, basis_order, dx_order)::Array{Float64, 4}
    function fast_sub(start, n)
        # Benching showed that this was way faster than any sort of array stuff
        # no allocations
        temp = 1
        for i=(start-n+1):(start)
            temp *= i
        end
        temp
    end
    function dpx(x, base_exp, n_derivs)
        # calculate derivative of x^base_exp
        if (n_derivs == 0)
            return x ^ base_exp
        end
        if (base_exp < n_derivs)
            return 0
        end
        coef = fast_sub(base_exp, n_derivs)
        exp = base_exp - n_derivs
        return coef * (x ^ exp)
    end
    function dHx!(x_x_I, order, n_d, Hx)
        for h=0:n_d
            i_d = n_d-h
            j_d = h
            k = 1
            for i=0:order
                for j=0:(order-i)
                    xi = dpx(x_x_I[1], i, i_d)
                    xj = dpx(x_x_I[2], j, j_d)
                    Hx[k,h+1] = xi*xj
                    k += 1
                end
            end
        end
    end
    if (dx_order == 0)
        Hx = get_Hx_from_diff_mat(x_x_I, basis_order)
        Hxs = size(Hx)
        Hx = reshape(Hx, size(Hx,1), 1, size(Hx,2), size(Hx,3))
        return Hx
    end
    Hx_size = get_Hx_size(basis_order, 2)
    n_eval = size(x_x_I, 2)
    n_pts = size(x_x_I, 3)
    Hx = zeros(Hx_size, dx_order+1, n_eval, n_pts)
    Threads.@threads for j=1:n_pts
        for i=1:n_eval
            @views dHx!(x_x_I[:,i,j], basis_order, dx_order, Hx[:,:, i, j])
        end
    end
    Hx
end

function get_diff_matrix(x, x_I)
    x_I_r = reshape(x_I,2,1,size(x_I,2))
    x_I_l = reshape(x,2,size(x,2),1)
    x_I_l .- x_I_r
end

function get_coefficients(x_x_I, Hx_x_I, phi_a, order)
    n_eval = size(x_x_I, 2)
    Hx_size = size(Hx_x_I, 1)
    B = zeros(Hx_size, n_eval)
    H0 = zeros(Hx_size)
    H0[1] = 1
    Threads.@threads for i=1:n_eval
        # get moment matrix
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
    psi = zeros(n_eval, n_pts)
    Threads.@threads for j=1:n_eval
        for i=1:n_pts
            # eq (5.10)
            psi[j,i] = B[:,j]' * Hx_x_I[:, j, i] .* phi_a[j,i]
        end
    end
    psi
end


function rkpm_shape_funcs(x::Matrix{Float64}, x_i::Matrix{Float64}, order::Int, a::Float64)
    x_x_I = get_diff_matrix(x, x_i)
    Hx_x_I = get_Hx_from_diff_mat(x_x_I, order)
    phi_a = basis(x_x_I, a)
    B = get_coefficients(x_x_I, Hx_x_I, phi_a, order)
    calc_psi(B, Hx_x_I, phi_a)
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
domain_sample = domain[1:2:end]

# our test function
f = (x,y) -> x + cos(x * pi) + sin(y * pi)
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

psi = rkpm_shape_funcs(X, X_I, 3, a)
u_I = psi * Y_I
surface(X[1,:], X[2,:], u_I, legend=false)
