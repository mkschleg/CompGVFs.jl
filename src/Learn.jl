
function update!(lu, gvfs::Horde, args...)
    # We should thunk here....
    # @info gvfs
    for i in 1:length(gvfs)
	update!(lu, gvfs[i], args...)
    end
end

struct TDλ
    α::Float32
    λ::Float32
end

# Like optimizers?
function initialize(::TDλ, gvf::GVF)
    #How do we initialize?
end

function update!(lu, gvf::GVF, o_t, x_t, a_t, μ_t, o_tp1, x_tp1, r_tp1, p_tp1)
	
    ρ_t = if gvf.π isa OnPolicy
       	one(eltype(gvf.w))
    else
        get_value(gvf.π, o_t, a_t)/μ_t
    end
	
    γ_t, γ_tp1 = if gvf.γ isa AbstractFloat
        eltype(w)(gvf.γ)
    else
        get_value(gvf.γ, o_t, x_t), get_value(gvf.γ, o_tp1, x_tp1)
    end

    c = get_value(gvf.c, o_tp1, x_tp1, p_tp1, r_tp1)

    update!(lu, gvf, x_t, x_tp1, ρ_t, c, γ_t, γ_tp1)
end

function update!(lu::TDλ, 
		 gvf,
		 x_t, x_tp1, 
		 ρ_t, c, γ_t, γ_tp1)
    
    λ = lu.λ
    w = gvf.w
    z = gvf.z
	
    δ = c + γ_tp1*predict(gvf, x_tp1) - predict(gvf, x_t)


    if eltype(x_t) <: Integer
        # Tile Coded features
        z .*= γ_t*λ
        view(z, x_t) .+= 1
        z .*= ρ_t
        w .+= (lu.α * δ) .* z
    else
        # Floating Point
        z .= ρ_t .* ((γ_t*λ) .* gvf.z .+ x_t)
        w .+= lu.α * δ * gvf.z
    end
end

# ╔═╡ 47eba562-99e0-4815-a028-21f7376cd257
function update!(lu::TDλ,
		 gvf,
		 x_t::Int, x_tp1::Int, 
		 ρ_t, c, γ_t, γ_tp1)
    
    λ = lu.λ

    w = gvf.w
    z = gvf.z
	
    δ = c + γ_tp1*predict(gvf, x_tp1) - predict(gvf, x_t)

    
    z .*= γ_t*λ
    view(z, x_t) .+= 1
    z .*= ρ_t
    w .+= (lu.α * δ) .* z

end

struct Qλ
	α::Float32
	λ::Float32
end

function update!(lu::Qλ, 
	         gvf::BDemon, 
	         o_t, x_t, 
	         a_t, μ_t, o_tp1, x_tp1, r_tp1, p_tp1)

    γ_t, γ_tp1 = if gvf.γ isa AbstractFloat
        eltype(w)(gvf.γ)
    else
        get_value(gvf.γ, o_t, x_t), get_value(gvf.γ, o_tp1, x_tp1)
    end

    c = get_value(gvf.c, o_tp1, x_tp1, p_tp1, r_tp1)

	update!(lu, gvf, x_t, x_tp1, a_t, c, γ_t, γ_tp1)
end

function update!(
		lu::Qλ, 
		bdemon,
		x_t::Int, x_tp1::Int, 
		a_t::Int, c, γ_t, γ_tp1)
    
    λ = lu.λ

    w = bdemon.w
    z = bdemon.z

    Q_t = predict(bdemon, a_t, x_t)
    Q_tp1 = maximum(predict(bdemon, x_tp1))
	
    δ = c + γ_tp1*Q_tp1 - Q_t
    
    z .*= γ_t*λ
    view(z, a_t, x_t) .+= 1
    w .+= (lu.α * δ) .* z

end

# function MinimalRLCore.is_terminal(bdemon::BDemon, s_t, x_t)
#     get_value(bdemon.γ, s_t, x_t) == 0
# end

