using LinearAlgebra
using Plots

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

function jitter!(X, scale)
    X += rand(size(X)...) .* scale
end

step_size = .2
domain = -2:step_size:2
domain_sample = domain[1:1:end]
prod_d = x-> Iterators.product(x, x)

# our test function
f = (x,y) -> cos(x * pi ) + sin(y * pi)
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
basis = x -> bspline.(sqrt.(sum(x.^2,dims=3))/a)/a

#x-x_I matrix, x_x_I[j,i] is x-x_I at point x=x_j
x_x_I = zeros(n_eval, n_pts, 2)
for i=1:2
    x_x_I[:,:,i] = X[:,i] .- X_I[:,i]'
end
phi_a = basis(x_x_I)
order = 4
Hx_size = Int((order+1)*(order+2)/2)
Mx = zeros(Hx_size, Hx_size, n_eval)
for i=1:n_eval
    Hx = Mx[:, :, i]
    for j=1:n_pts
        if phi_a[i,j] <= 1e-10
            continue
        end
        H_np = get_Hx(x_x_I[i, j, :], order)
        H_np2 = H_np * H_np'
        Hx += H_np2 .* phi_a[i,j]
    end
    Mx[:, :, i] = Hx
end

B = zeros(Hx_size, n_eval)
H0 = zeros(Hx_size)
H0[1] = 1
for i=1:n_eval
    # eq. 5.9
    B[:,i] = Mx[:,:,i] \ H0
end

# get kronecker 
psi = zeros(n_pts, n_pts)
for i=1:n_pts
    for j=1:n_eval
        # eq (5.10)
        psi[j,i] = B[:,j]' * get_Hx(x_x_I[j, i, :], order) .* phi_a[j,i]
    end
end

u_h = psi * Y_I
psi_inv = inv(psi)
u_I = psi_inv * u_h


#surface(X[:,1], X[:,2], u_h[:,1])
surface(X[:,1], X[:,2], (u_I[:,1]-Y), legend=false)
@show sum(abs.(u_h-Y))/sum(abs.(Y))
#p=scatter(X[:,1], X[:,2], zeros(length(X[:,1])))
