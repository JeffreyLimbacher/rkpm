using LinearAlgebra
using Plots


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
basis = x -> bspline(sqrt(sum((x).^2))/a)

mutable struct Rkpm
    x::Matrix{Float64}
    psi::Matrix{Float64}
    A::Matrix{Float64}
    order::Float64
end

function get_B(x_I::Matrix{Float64}, x::Matrix{Float64})
    x_x_I = x .- x'
end

function init_rkpm(x::Matrix{Float64}, y, order)
    x_x_I = x .- x'
    n_pts = size(x,1)
    phi_a = basis.(x_x_I)
    order = 2
    Mx = zeros(order+1, order+1, n_eval)
    for i=1:n_pts
        Hx = Mx[:, :, i]
        for j=1:n_pts
            H_np = x_x_I[i, j].^(0:order)
            H_np2 = H_np * H_np'
            Hx += H_np2 .* phi_a[i,j]
        end
        Mx[:, :, i] = Hx
    end

    B = zeros(order+1, n_eval)
    H0 = zeros(order + 1)
    H0[1] = 1
    for i=1:n_pts
        B[:,i] = Mx[:,:,i] \ H0
    end

    #eval at each eval pt
    psi = zeros(n_eval, n_pts)
    u_h_I = similar(psi)
    for i=1:n_pts
        for j=1:n_eval
            psi[j,i] = B[:,j]' * x_x_I[j, i].^(0:order) .* phi_a[j,i]
            u_h_I[j,i] = psi[j,i]*Y_I[i]
        end
    end

    A = inv(psi)
    rkpm(
        x,
        psi,
        A,
        order
    )
end

function Rkpm(rkpm::Rkpm, points::Matrix{Float64})
    n_eval = size(points, 1)

end

inc = 0.2
X_I = collect(-3:inc:3)
X_I = X_I + rand(length(X_I)) .* inc / 2.0
Y_I = sin.(X_I)
n_pts = length(X_I)
vals = ones(Float64, length(X_I))

X = -3:.01:3
n_eval = length(X)


u_h = sum(u_h_I, dims=2)
p=plot(X,[u_h_I, u_h],legend=false)
plot!(p, X_I, Y_I, linestyle=:dash,lw=2)
scatter!(p, X_I, zeros(length(X_I)), )

# plot consistency conditions
#plot(X, [sum(psi,dims=2), sum(psi.*X_I',dims=2)])
