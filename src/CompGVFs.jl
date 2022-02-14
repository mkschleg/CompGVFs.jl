module CompGVFs

# Write your package code here.

# ╔═╡ ff6da4df-5c4f-4167-9fe9-e965d63165bf
struct FeatureCumulant
    idx::Int
end

# ╔═╡ 118bac06-bd77-4ab3-a5fe-f9a7590cb781
get_value(fc::FeatureCumulant, o, x, p, r) = x[fc.idx]

# ╔═╡ 289d39df-98c3-4572-8347-7ecf704f4be5
get_value(fc::FeatureCumulant, o, x::Vector{Int}, p, r) = fc.idx ∈ x

# ╔═╡ 285488f4-1e74-4e15-816b-077722c4677b
struct ObservationCumulant
    idx::Int
end

# ╔═╡ 0dbae625-1a58-4fc9-a115-84411becdcc0
get_value(oc::ObservationCumulant, o, x, p, r) = o[oc.idx]

# ╔═╡ 9f03c3bc-6fac-4e5d-8b39-7e0b5a891e71
struct PredictionCumulant
	idx::Int
end

# ╔═╡ 9ce73bf3-1cdb-4a0f-bbab-b7c331a7b7fe
get_value(pc::PredictionCumulant, o, x, p, r) = p[pc.idx]

# ╔═╡ 3dbe0930-8fb7-4dea-b2e3-0110a28a7249
struct RescaleCumulant{C, F}
	c::C
	γ::F
end

# ╔═╡ ab28215c-74c0-4722-b96d-5c3862dab13d
get_value(rsc::RescaleCumulant, o, x, p, r) = 
	get_value(rsc.c, o, x, p, r)*(1-rsc.γ)

# ╔═╡ 6b865983-65d2-478f-b891-bdaf0092e2ce
struct ThresholdCumulant{C, F}
	c::C
	θ::F
end

# ╔═╡ f2ab0915-4752-4248-bb65-d90b3f929539
get_value(rsc::ThresholdCumulant, o, x, p, r) = 
	get_value(rsc.c, o, x, p, r) >= rsc.θ ? 1 : 0

# ╔═╡ 131b2d9b-711b-4cea-bab1-03f0ef68f5a9
struct OnPolicy end

# ╔═╡ 91fdbc6f-479c-4a79-848b-b0a83268348b
get_value(op::OnPolicy, args...) = 1.0

# ╔═╡ 725c9586-615d-4d27-8a2f-fe2760aeaedc
struct ConstantDiscount{F<:AbstractFloat}
	γ::F
end

# ╔═╡ 7cbb8f85-1fd3-4c0d-a081-0d9c487227e6
get_value(cd::ConstantDiscount, args...) = cd.γ

# ╔═╡ 4d7bcae2-e7dd-4aa4-84e5-6c529be7c2b4
struct TerminatingDiscount{F<:AbstractFloat}
	γ::F
	idx::Int
end

# ╔═╡ 85392583-6481-4a77-96c0-30f136e08299
get_value(fc::TerminatingDiscount, o, x::Vector{Int}) = fc.idx ∈ x ? zero(typeof(fc.γ)) : fc.γ

# ╔═╡ 2410a3a2-f1d6-4edf-ae52-66d43816093b
# What are we doing.
# Massively parallel GVF learning
begin
	struct GVF{W, Z, C, Π, Γ}
		w::W
		z::Z
		c::C
		π::Π
		γ::Γ
	end
	GVF(num_feats, cumulant, policy, discount) = 
		GVF(zeros(num_feats), zeros(num_feats), cumulant, policy, discount)
end

# ╔═╡ 3d3dd633-4f9f-4cad-8018-dfe2bebffa5b
predict(gvf::GVF{<:AbstractVector}, x) = dot(gvf.w, x)

# ╔═╡ b40e514c-04d0-4690-ad89-b81761634bf4
predict(gvf::GVF{<:AbstractVector}, x::AbstractVector{Int}) = begin; 
	w = gvf.w; 
	@tullio ret := w[x[i]]; 
end

# ╔═╡ 0a9dbac5-195b-4c78-a0e4-62f6727e4fce
const Horde = Vector{<:GVF}

# ╔═╡ 0ce6eaec-1c2f-478b-96fd-fe4965517e46
predict(horde::Vector{<:GVF}, x) = [predict(gvf, x) for gvf in horde]

# ╔═╡ ae9413e4-918e-4405-928d-85d64bcebd17
struct TDλ
    α::Float32
    λ::Float32
end

# ╔═╡ adf1838e-bc3a-44de-baaa-de33e77c3296
function update!(
		lu::TDλ, 
		gvf,
		x_t, x_tp1, 
		ρ_t, c, γ_t, γ_tp1)
    
    λ = lu.λ

	w = gvf.w
	z = gvf.z
	
    δ = c + γ_tp1*predict(gvf, x_tp1) - predict(gvf, x_t)
    
    if eltype(x_t) <: Integer
        z .*= γ_t*λ
        # view(z, x_t) .+= 1
        @tullio z[x_t[i]] = z[x_t[i]] + 1
        z .*= ρ_t
        w .+= (lu.α * δ) .* z
    else
        z .= ρ_t .* ((γ_t*λ) .* gvf.z .+ x_t)
        w .+= lu.α * δ * gvf.z
    end
end

# ╔═╡ 66d0475a-cfd5-4805-b79f-298645fe9592
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

# ╔═╡ 83128a03-0274-432d-b307-960d65502dae
function update!(lu, gvfs::Vector{G}, args...) where G<:GVF
    Threads.@threads for i in 1:length(gvfs)
		update!(lu, gvfs[i], args...)
    end
end



end
