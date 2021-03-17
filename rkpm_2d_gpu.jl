using CUDA
using LinearAlgebra
using Plots

function get_domain(x_pts)
    np = length(x_pts)
    x = CUDA.zeros(np, np, 2)
    x[:, :, 1] .= x_pts
    x[:, :, 2] .= x_pts'
    reshape(x, np * np, 2)
end

#x_x_I is the vector of differences between the points
function get_Hx(x_x_I, order)
    exps = v[(i, j) for i=0:order for j=0:order if i+j<=order]
    hx = zeros(length(exps))
    for (i,exp) in enumerate(exps)
        hx[i] = prod(x_x_I.^exp)
    end
    hx
end

function hx_kernel(x_x_I, H_x, order)
    idxx = blockDim().x *  (blockIdx().x - 1) + threadIdx().x
    idxy = blockDim().y *  (blockIdx().y - 1) + threadIdx().y
    sx = blockDim().x * gridDim().x
    sy = blockDim().x * gridDim().y
    for ix = idxx:sx:size(x_x_I, 1)
        for iy = idxy:sy:size(x_x_I, 2)
            id = 1
            for x=0:order
                for y=0:(order - x)
                    H_x[id,ix,iy] = CUDA.pow(x_x_I[ix,iy,1],x) * CUDA.pow(x_x_I[ix,iy,2],y)
                    id += 1
                end
            end
        end
    end
    nothing
end

function get_Hx_from_diff_mat(x_x_I, order)
    Hx_size = Int((order+1)*(order+2)/2)
    n_eval = size(x_x_I, 1)
    n_pts = size(x_x_I, 2)
    Hx = CUDA.zeros(Hx_size, n_eval, n_pts)
    CUDA.@cuda threads=(16,16) hx_kernel(x_x_I, Hx, order)
    Hx
end

function get_diff_matrix(x_I, x)
    n_pts = size(x, 1)
    n_eval = size(x_I, 1)
    x_I_r = reshape(X_I, 1,size(X_I,1),2)
    x_I_l = reshape(X_I,size(X_I,1),1,2)
    x_x_I = x_I_l .- x_I_r
end

function get_coefficients(x_x_I, Hx_x_I, phi_a, order)
    n_eval = size(x_x_I, 1)
    Hx_size = size(Hx_x_I, 3)
    B = zeros(Hx_size, n_eval)
    H0 = zeros(Hx_size)
    H0[1] = 1
    for i=1:n_eval
        Mx = zeros(Hx_size, Hx_size)
        for j=1:n_pts
            H_np = Hx_x_I[i, j, :]
            H_np2 = H_np * H_np'
            Mx += H_np2 .* phi_a[i,j]
        end
        # eq. 5.9
        B[:,i] = Mx \ H0
    end
    B
end

function cu_get_coefficients(x_x_I, Hx_x_I, phi_a, order)
    n_eval = size(phi_a, 1)
    n_pts = size(phi_a, 2)
    Hx_size = size(Hx_x_I, 1)
    # reshape for gemm_strided_batch (strided outer product)
    Hx_x_Iband = reshape(Hx_x_I, Hx_size, 1, n_pts*n_eval)
    # reshape for scalar broadcast
    phi_a_exp = reshape(phi_a, 1,1,n_eval,n_pts)
    temp = reshape(CUBLAS.gemm_strided_batched('N','T',1.0f0,Hx_x_Iband,Hx_x_Iband), Hx_size, Hx_size, n_pts, n_pts).*phi_a_exp
    # get the resulting M matrices ([:,:,i] is ith M matrix)
    temp = dropdims(sum(temp, dims=3), dims=(3,))
    H0 = CUDA.zeros(size(temp, 2),1)
    H0[1,1] = 1

    B = CUDA.zeros(Hx_size, n_pts)
    for i=1:n_eval
        B[:,i] = temp[:,:,i] \ H0
    end
    B
end

function get_psi(Hx_x_I, B, phi_a)
    n_eval = size(phi_a, 1)
    n_pts = size(phi_a, 2)

    phi_a_b = reshape(phi_a, 1, n_eval, n_pts)
    Hx_x_I_phi = Hx_x_I .* phi_a_b
    B_rowed = reshape(B, 1, size(B,1), size(B,2))
    psi = CUBLAS.gemm_strided_batched('N','N',1.0f0,B_rowed, Hx_x_I_phi)
    dropdims(psi;dims=1)
end

step_size = .1
domain = 0:step_size:5
domain_sample = domain[1:2:end]

# our test function
f = (x,y) -> CUDA.cos(x * pi) + CUDA.sin(y * pi)
X_I = get_domain(domain)
n_pts = size(X_I,1)
Y_I = f.(X_I[:,1], X_I[:,2])
X = get_domain(domain)
Y = f.(X[:,1], X[:,2])
n_eval = size(X,1)

#n_pts is number of interpolation points
#n_eval is true function and where evaluate interpolation
# support size
a = 0.3 .* (X_I[end]-X_I[1])

function bspline(z::T)::T where {T}
    zabs = abs(z)
    if zabs <= 0.5
        2.0/3.0 - 4z^2 + 4z^3
    elseif zabs <= 1.0
        4.0/3.0 * (1-z)^3
    else
        0
    end
end

function basis(x, a::Float32)
    z=sqrt.(sum(CUDA.pow.(x,2),dims=3))./a
    z=CUDA.map(bspline, z)
    z = z ./ a
    dropdims(z,dims=3)
end

# Get psi following RK method
x_x_I = get_diff_matrix(X_I, X_I)
Hx_x_I = get_Hx_from_diff_mat(x_x_I, 3)
phi_a = basis(x_x_I, Float32(a))
B = cu_get_coefficients(x_x_I, Hx_x_I, phi_a, 3)
psi = get_psi(Hx_x_I, B, phi_a)

u_h = psi * Y_I
surface(X[:,1], X[:,2], u_h[:,1])
