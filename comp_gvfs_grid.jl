### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 7bc2d377-7c94-4cee-8e3a-e8ce50a471b3
using Random

# ╔═╡ f6a92f82-be00-11ed-14fc-450478eca211
using MinimalRLCore

# ╔═╡ fd86d35c-9433-4e68-88b8-078c56bcf2a4
using PlutoUI, ProgressLogging

# ╔═╡ 94b20a16-cccf-4512-967d-77261222544b
using Plots

# ╔═╡ 2bf20b76-74e2-42b2-8053-fa66657c5563
using RecipesBase, Colors

# ╔═╡ 3b85f9cd-f9aa-47ac-852d-ed7839167790
using Statistics

# ╔═╡ e09f74b0-fff2-4417-8867-f05f47c6a055
PlutoUI.TableOfContents(title="Comp-GVFS", indent=true, aside=true)

# ╔═╡ a155572b-9273-491d-bcbe-9540efd09081
color_scheme = [
    colorant"#44AA99",
    colorant"#332288",
    colorant"#DDCC77",
    colorant"#999933",
    colorant"#CC6677",
    colorant"#AA4499",
    colorant"#DDDDDD",
	colorant"#117733",
	colorant"#882255",
	colorant"#1E90FF",
]

# ╔═╡ 4dab38f3-a07f-46e5-8bcd-26132c65ca43
md"""
# GVF details
"""

# ╔═╡ 5ab59373-6a80-46b2-8687-ade15aa31b5e
md"""
## Parameter Functions
"""

# ╔═╡ c9f0f1e2-d596-45d4-9a6b-ead881e615f3
md"""
### Cumulants
"""

# ╔═╡ 440a6393-d3f2-4158-9cad-a9a3ffb658b7
struct FeatureCumulant
    idx::Int
end

# ╔═╡ 49c35c5b-e99c-4768-99fd-6344f8ebd3e1
get_value(fc::FeatureCumulant, o, x, p, r) = x[fc.idx]

# ╔═╡ e3a469fb-98d1-4a17-b1f4-2dc77a8224a4
get_value(fc::FeatureCumulant, o, x::Int, p, r) = fc.idx == x

# ╔═╡ 3f3048ff-843f-41b0-aa86-b64090cddc57
get_value(fc::FeatureCumulant, o, x::Vector{Int}, p, r) = fc.idx ∈ x

# ╔═╡ a22330b9-c3f7-42da-929b-1023ccea6df2
struct ObservationCumulant
    idx::Int
end

# ╔═╡ ca873653-1b35-42a7-a3f6-b0b6f8c3d314
get_value(oc::ObservationCumulant, o, x, p, r) = o[oc.idx]

# ╔═╡ 0c583b98-3740-4812-9969-e2d00d03ce3d
struct PredictionCumulant
	idx::Int
end

# ╔═╡ 8abd1d2e-6519-4352-9859-4ed1f3bfc1d0
get_value(pc::PredictionCumulant, o, x, p, r) = p[pc.idx]

# ╔═╡ 184ecfc1-127f-4585-a618-3d159f79d718
struct RescaleCumulant{C, F}
	c::C
	γ::F
end

# ╔═╡ 3527021b-de33-42da-aac0-619483caba3e
get_value(rsc::RescaleCumulant, o, x, p, r) = 
	get_value(rsc.c, o, x, p, r)*(1-rsc.γ)

# ╔═╡ 713bdb3f-7afe-4215-9191-b7704383c6b0
struct ThresholdCumulant{C, F}
	c::C
	θ::F
end

# ╔═╡ 73ae37d9-2a85-4e3b-b2b0-29b9b0cbc029
get_value(rsc::ThresholdCumulant, o, x, p, r) = 
	get_value(rsc.c, o, x, p, r) >= rsc.θ ? 1 : 0

# ╔═╡ b8130b16-52c9-42e2-8e87-03e047547809
md"""
### Policies
"""

# ╔═╡ 2756de72-2851-43cd-a058-da96d294b70c
struct OnPolicy end

# ╔═╡ 0939ea40-1170-4269-9e6e-6ba5a0c80d72
get_value(op::OnPolicy, args...) = 1.0

# ╔═╡ b1bc29d3-9790-47fc-9ca5-f8215ffa1188
md"""
### Discount
"""

# ╔═╡ dfb951c9-5bfa-4307-864f-af02c5c1578d
struct ConstantDiscount{F}
	γ::F
end

# ╔═╡ 6d1f2c5c-90e5-41e7-a6b9-1abb7f2426ad
get_value(cd::ConstantDiscount, args...) = cd.γ

# ╔═╡ 2e621539-afc9-4306-8f2b-d418099f5a34
struct TerminatingDiscount{F}
	γ::F
	idx::Int
end

# ╔═╡ 7bbe957e-0669-42e5-9384-e31fc31f8283
get_value(fc::TerminatingDiscount, o, x::Vector{Int}) = fc.idx ∈ x ? zero(typeof(fc.γ)) : fc.γ

# ╔═╡ c2a35607-bf5c-441f-a708-53f32101f33e
get_value(fc::TerminatingDiscount, o, x::Int) = fc.idx == x ? zero(typeof(fc.γ)) : fc.γ

# ╔═╡ 6481cbd3-7ae0-4adc-b7bd-d4fa1957defa
md"""
## GVF
"""

# ╔═╡ d6e1b5c1-568d-43d5-a90f-dd0dd3c907f7
# What are we doing.
# Massively parallel linear GVF learning
begin
	struct GVF{W, C, Π, Γ}
		w::W
		z::W
		c::C
		π::Π
		γ::Γ
	end
	GVF(num_feats, cumulant, policy, discount) = 
		GVF(zeros(num_feats), zeros(num_feats), cumulant, policy, discount)
end

# ╔═╡ 5e6ded46-1ff1-4667-966c-7e2bd4a35945
struct GVFCumulant{G<:GVF}
	gvf::G
end

# ╔═╡ 51e69bed-bf48-4688-87fd-d0c9e9bfc34e
struct GVFThreshTerminatingDiscount{F, G<:GVF}
	γ::F
	gvf::G
	θ::Float64
end

# ╔═╡ d135f4fe-8700-4a60-acec-890a9dc562fc
mutable struct GVFThreshTerminatingMaxDiscount{F, G<:GVF}
	γ::F
	gvf::G
	θ::Float64
	GVFThreshTerminatingMaxDiscount(γ, gvf) = new{typeof(γ), typeof(gvf)}(γ, gvf, -Inf)
end

# ╔═╡ 0ba53300-2ff7-4c59-b7c9-d9896ed497af
predict(gvf::GVF{<:AbstractVector}, x::AbstractVector{<:Number}) = 
	dot(gvf.w, x)

# ╔═╡ 4e01a022-5bb6-47de-81d3-9ff418198f6c
predict(gvf::GVF{<:AbstractVector}, x::AbstractVector{Int}) = begin; 
	w = gvf.w; 
	ret = sum(view(w, x))
end

# ╔═╡ 8a44f9bd-b7ab-4102-8234-d73f7e979fa9
predict(gvf::GVF{<:AbstractVector}, x::Int) = begin; 
	gvf.w[x]
end

# ╔═╡ 594b5d03-9dc9-4df7-bef4-a71a81224e27
predict(gvf::GVF{<:AbstractMatrix}, x::AbstractVector{Int}) = begin; 
	gvf.w[x]
end

# ╔═╡ b8e3c48e-e7f6-402f-8f44-08663bd273a2
md"""
## Horde
"""

# ╔═╡ b4c75f1b-fbc5-4370-8241-f51e7865fcdd
const Horde = Vector{<:GVF}

# ╔═╡ 7b3b895a-4563-42be-8d4f-5cb9dd53a6d7
predict(horde::Horde, x) = [predict(gvf, x) for gvf in horde]

# ╔═╡ 32c22f6c-02b2-4064-9b79-eeed1bb975ce
struct TDλ
    α::Float32
    λ::Float32
end

# ╔═╡ a69fed13-737d-4fd6-8baf-0dabe7f339a5
function update!(lu, gvfs::Vector{G}, args...) where G<:GVF
	# We should thunk here....
	# @info gvfs
	for i in 1:length(gvfs)
		update!(lu, gvfs[i], args...)
    end
end

# ╔═╡ 356c80aa-d825-45dc-8708-cf7ad6ede14f
md"# Learned Policies"

# ╔═╡ 217a9a10-1b57-4d4a-8b41-eeaea71c749f
struct ϵGreedy
	ϵ::Float32
end

# ╔═╡ 5001c4a2-343d-4f5e-a2e4-4e01750932e9
function get_value(π::ϵGreedy, q, a_t)
	idx = findall(==(maximum(q)), q)
	if a_t ∈ idx
		(1 - π.ϵ) / length(idx)
	else
		π.ϵ / length(q)
	end
end

# ╔═╡ 695fd511-5f59-4e77-8b26-5e3787dbc198
findall(==(maximum([1, 1, 0, 0])), [1, 1, 0, 0])

# ╔═╡ 708a85fd-9b52-420b-a468-f8b4129ce794
function get_action(rng::Random.AbstractRNG, π::ϵGreedy, q)
	if rand() < π.ϵ
		rand(rng, 1:length(q))
	else
		rand(rng, findall(==(maximum(q)), q))
	end
end

# ╔═╡ f7405d98-10e9-4e2e-ac13-1780dc6c5446
get_action(π, q) = get_action(Random.default_rng(), π, q)

# ╔═╡ ba9085c6-cecb-4ea5-8f3d-0ed21a23c196
mutable struct BDemon{W, C, Π, Γ}
	w::W
	z::W
	c::C
	π::Π
	γ::Γ
	BDemon(num_feats::Int, num_actions::Int, cumulant, policy, discount) = 
		new{Matrix{Float64}, typeof(cumulant), typeof(policy), typeof(discount)}(zeros(num_actions, num_feats), zeros(num_actions, num_feats), cumulant, policy, discount)
end

# ╔═╡ cda934f1-08f7-460a-b4b2-c9c20ade52a4
function MinimalRLCore.start!(bdemon::BDemon, s_t, x_t=s_t)
	bdemon.z .= 0
end

# ╔═╡ 6733a1bb-88bd-4e01-9a03-dd39279ba1e7
predict(bdemon::BDemon{<:AbstractMatrix}, x::Int) = begin; 
	bdemon.w[:, x]
end

# ╔═╡ 33db32af-2e34-4dad-abcc-e023349aae70
predict(bdemon::BDemon{<:AbstractMatrix}, a::Int, x::Int) = begin; 
	bdemon.w[a, x]
end

# ╔═╡ 349185f3-855d-40b7-a669-8468e0bb686c
get_value(gvfc::GVFCumulant, o, x, p, r) = 
	predict(gvfc.gvf, x)

# ╔═╡ fa854b1a-a28c-4a28-b8e4-3e03a4a99810
get_value(fc::GVFThreshTerminatingDiscount, o, x::Int) = begin
	# if predict(fc.gvf, x) > fc.θ
	# 	# @info "terminante"
	# end
	predict(fc.gvf, x) > fc.θ ? zero(typeof(fc.γ)) : fc.γ
	# fc.idx == x ? zero(typeof(fc.γ)) : fc.γ
end

# ╔═╡ 26a2fd60-0812-404b-a1d5-145a24d6fe35
get_value(fc::GVFThreshTerminatingMaxDiscount, o, x::Int) = begin
	pred = predict(fc.gvf, x)
	if pred > fc.θ || pred ≈ fc.θ
		# @info "terminate"
		fc.θ = pred
		return zero(typeof(fc.γ))
	else
		return fc.γ
	end
end

# ╔═╡ 0398417f-db77-4ec7-b385-9f3206a5365c
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
        view(z, x_t) .+= 1
        z .*= ρ_t
        w .+= (lu.α * δ) .* z
	else
        z .= ρ_t .* ((γ_t*λ) .* gvf.z .+ x_t)
        w .+= lu.α * δ * gvf.z
    end
end

# ╔═╡ 47eba562-99e0-4815-a028-21f7376cd257
function update!(
		lu::TDλ, 
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

# ╔═╡ 660e8864-012f-4347-abf0-777d8c20aecd
function get_action(demon::BDemon, x_t)
	get_action(demon.π, predict(demon, x_t))
end

# ╔═╡ 1b9ec5be-ca0f-4221-abf1-17f44b5ba118
function get_action(rng::Random.AbstractRNG, demon::BDemon, x_t)
	get_action(rng, demon.π, predict(demon, x_t))
end

# ╔═╡ 3bffcae1-1570-4d9f-814a-702afd51d3e4
struct Qλ
	α::Float32
	λ::Float32
end

# ╔═╡ a351143c-a221-4c44-94a6-249308c38a11
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

# ╔═╡ 55a73859-6da8-4d4f-a391-a9bf63a2f98c
function get_value(bdemon::BDemon, s_t, a_t)
	get_value(bdemon.π, predict(bdemon, s_t), a_t)
end

# ╔═╡ 44bba6f3-5f82-4c9a-b4da-e33f23010ee3
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

# ╔═╡ 3cff48e4-ad25-4354-8cdb-877586b33e96
function MinimalRLCore.is_terminal(bdemon::BDemon, s_t, x_t)
	get_value(bdemon.γ, s_t, x_t) == 0
end

# ╔═╡ 204234d7-212a-4985-a45d-1779286760e1
function update!(
	lu::Qλ, 
	gvf::BDemon, 
	o_t, x_t, 
	a_t, μ_t, o_tp1, x_tp1, r_tp1, p_tp1)
	
	# ρ_t = if gvf.π isa OnPolicy
 #       	one(eltype(gvf.w))
 #    else
 #        get_value(gvf.π, o_t, a_t)/μ_t
 #    end
	
    γ_t, γ_tp1 = if gvf.γ isa AbstractFloat
        eltype(w)(gvf.γ)
    else
        get_value(gvf.γ, o_t, x_t), get_value(gvf.γ, o_tp1, x_tp1)
    end

    c = get_value(gvf.c, o_tp1, x_tp1, p_tp1, r_tp1)

	update!(lu, gvf, x_t, x_tp1, a_t, c, γ_t, γ_tp1)
end

# ╔═╡ 0bc9f5c5-6fb4-4874-8820-3345f50055e6
md"""
# Environments
"""

# ╔═╡ dd006581-f89c-426c-9ade-7f7edb58dc88
md"""
## Cycle World
"""

# ╔═╡ 981a4a12-8f59-4d8a-b734-1993280959e6
mutable struct CycleWorld <: AbstractEnvironment
	chain_length::Int64
	agent_state::Int64
	partially_observable::Bool
	CycleWorld(chain_length::Int64; 
			   rng=Random.GLOBAL_RNG, 
			   partially_observable=true) =
		new(chain_length,
			0,
			partially_observable)
end

# ╔═╡ d864d151-5d71-4851-a8ec-72e415573476
function MinimalRLCore.reset!(env::CycleWorld, rng=nothing; kwargs...)
	env.agent_state = 0
end

# ╔═╡ c0c9cf24-7294-4e26-8d0f-f978c9996e04
MinimalRLCore.get_actions(env::CycleWorld) = Set(1:1)

# ╔═╡ e52b2a09-e4f7-4756-96a8-6dce21a3eaac
function MinimalRLCore.environment_step!(
			env::CycleWorld,
			action, 
			rng=nothing; 
			kwargs...)
	env.agent_state = (env.agent_state + 1) % env.chain_length
end

# ╔═╡ 4bffba42-2d29-46ac-9bdd-59f348d92a30
MinimalRLCore.get_reward(env::CycleWorld) = 0 # -> get the reward of the environment

# ╔═╡ 7b266d56-2a4d-40de-9a9f-3a9680bebacc
fully_observable_state(env::CycleWorld) = [env.agent_state+1]

# ╔═╡ a2abdbd6-c2a4-443f-b724-6bf574436ee7
function partially_observable_state(env::CycleWorld)
	state = zeros(1)
	if env.agent_state == 0
		state[1] = 1
	end
	return state
end

# ╔═╡ 924b6945-3826-4138-aa00-206ddb9840a5
function partially_observable_state(state::Int)
	state = zeros(1)	
	if state == 0
		state[1] = 1
	end
	return state
end

# ╔═╡ 31115f9d-c987-4b11-9b02-ee7942606f7a
function MinimalRLCore.get_state(env::CycleWorld) # -> get state of agent
	if env.partially_observable
		return partially_observable_state(env)
	else
		return fully_observable_state(env)
	end
end

# ╔═╡ f18749cd-0160-44ed-b91f-30a6a9dfad8f
function MinimalRLCore.is_terminal(env::CycleWorld) # -> determines if the agent_state is terminal
	return false
end

# ╔═╡ 1775dc2a-91ea-4303-a5c7-b260dd06278d
function Base.show(io::IO, env::CycleWorld)
	model = fill("0", env.chain_length)
	model[1] = "1"
	println(io, join(model, ' '))
	model = fill("-", env.chain_length)
	model[env.agent_state + 1] = "^"
	println(io, join(model, ' '))
	# println(env.agent_state)
end

# ╔═╡ 940461dc-735f-4297-ad5d-f67a23ebbb5f
md"## FourRooms"

# ╔═╡ 1e62ec00-975b-492c-bb11-f7b7892a58e8
mutable struct GridWithWalls <: AbstractEnvironment
    state::Vector{Int}
    const walls::Array{Bool, 2}
	const ACTIONS::Tuple{Int, Int, Int, Int}
	const UP
	const RIGHT
	const DOWN
	const LEFT
	function GridWithWalls(size::Int, wall_list::Array{CartesianIndex{2}})
	    walls = fill(false, size[1], size[2])
	    for wall in wall_list
	        walls[wall] = true
	    end
	    new([1,1], walls)
	end
	function GridWithWalls(walls::Array{Int64, 2})
	    new([1,1], convert(Array{Bool}, walls), (1, 2, 3, 4), 1, 2, 3, 4)
	end
end

# ╔═╡ 70d5188f-20dd-49cd-a18d-aeefa16d8bb2
function FourRooms()
	BASE_WALLS = [0 0 0 0 0 1 0 0 0 0 0;
              	  0 0 0 0 0 1 0 0 0 0 0;
             	  0 0 0 0 0 0 0 0 0 0 0;
              	  0 0 0 0 0 1 0 0 0 0 0;
              	  0 0 0 0 0 1 0 0 0 0 0;
            	  1 0 1 1 1 1 0 0 0 0 0;
              	  0 0 0 0 0 1 1 1 0 1 1;
              	  0 0 0 0 0 1 0 0 0 0 0;
              	  0 0 0 0 0 1 0 0 0 0 0;
              	  0 0 0 0 0 0 0 0 0 0 0;
              	  0 0 0 0 0 1 0 0 0 0 0;]
	# BASE_WALLS = [0 0 0 0 0 0 0 0 0 0 0;
 #              	  0 0 0 0 0 0 0 0 0 0 0;
 #             	  0 0 0 0 0 0 0 0 0 0 0;
 #              	  0 0 0 0 0 0 0 0 0 0 0;
 #              	  0 0 0 0 0 0 0 0 0 0 0;
 #            	  0 0 0 0 0 0 0 0 0 0 0;
 #              	  0 0 0 0 0 0 0 0 0 0 0;
 #              	  0 0 0 0 0 0 0 0 0 0 0;
 #              	  0 0 0 0 0 0 0 0 0 0 0;
 #              	  0 0 0 0 0 0 0 0 0 0 0;
 #              	  0 0 0 0 0 0 0 0 0 0 0;]
	GridWithWalls(BASE_WALLS)
end

# ╔═╡ 621e43ea-2006-472e-b6a5-94ae68d07bf2
MinimalRLCore.is_terminal(env::GridWithWalls) = false

# ╔═╡ ab3597f3-ed93-42be-b22a-74c9a507cc57
MinimalRLCore.get_reward(env::GridWithWalls) = 0

# ╔═╡ f48d5184-2b3c-4fea-9948-66332e40d74d
is_wall(env::GridWithWalls, state) = env.walls[state[1], state[2]] == 1

# ╔═╡ 841a7a9f-fc8d-4107-a411-3039e1826325
Base.size(env::GridWithWalls, args...) = size(env.walls, args...)

# ╔═╡ 9b9bc9ae-5ea5-4f1d-afaf-7d4d20b2baf5
MinimalRLCore.get_state(env::GridWithWalls, state = env.state) = begin
	# @info state
	(state[1] - 1) * size(env, 1) + state[2]
end

# ╔═╡ 6bc7973a-98a2-40dd-8a2d-aa27887c8fa1
random_state(env::GridWithWalls, rng) = [rand(rng, 1:size(env.walls)[1]), rand(rng, 1:size(env.walls)[2])]

# ╔═╡ bdb1b1da-f2d3-4248-b5bd-d6135c61d89c
num_actions(env::GridWithWalls) = 4

# ╔═╡ b16223bf-e803-4c08-9d68-e688db7351e5
get_states(env::GridWithWalls) = findall(x->x==false, env.walls)

# ╔═╡ ab1bf706-9b9c-4b2f-b68b-65dfd6acf04b
MinimalRLCore.get_actions(env::GridWithWalls) = FourRoomsParams.ACTIONS

# ╔═╡ 07501680-edfd-4d17-b399-d37bb7806ee0
function MinimalRLCore.reset!(env::GridWithWalls, rng=nothing; kwargs...)
    state = random_state(env, rng)
    while env.walls[state[1], state[2]]
        state = random_state(env, rng)
    end
    env.state = state
    return state
end

# ╔═╡ 6f194e0d-e844-4d4c-b946-7ea8b75327fc
function MinimalRLCore.environment_step!(
	env::GridWithWalls, action, rng=Random.default_rng(); kwargs...)

    frp = env
    next_state = copy(env.state)

    if action == frp.UP
        next_state[1] -= 1
    elseif action == frp.DOWN
        next_state[1] += 1
    elseif action == frp.RIGHT
        next_state[2] += 1
    elseif action == frp.LEFT
        next_state[2] -= 1
    end

    next_state[1] = clamp(next_state[1], 1, size(env.walls, 1))
    next_state[2] = clamp(next_state[2], 1, size(env.walls, 2))
    if is_wall(env, next_state)
        next_state = env.state
    end

    env.state = next_state
end

# ╔═╡ 992f3a2d-a942-4d72-8f7d-c37f904e915f
function _step(env::GridWithWalls, state, action, rng, kwargs...)
    frp = env
    next_state = copy(state)

    if action == frp.UP
        next_state[1] -= 1
    elseif action == frp.DOWN
        next_state[1] += 1
    elseif action == frp.RIGHT
        next_state[2] += 1
    elseif action == frp.LEFT
        next_state[2] -= 1
    end

    next_state[1] = clamp(next_state[1], 1, size(env.walls, 1))
    next_state[2] = clamp(next_state[2], 1, size(env.walls, 2))
    if is_wall(env, next_state)
        next_state = state
    end

    return next_state, 0, false
end

# ╔═╡ 983c6296-db06-419c-81ed-eac04fa4db75
function _step(env::GridWithWalls, state::CartesianIndex{2}, action)
    array_state = [state[1], state[2]]
    new_state, r, t = _step(env, array_state, action)
    return CartesianIndex{2}(new_state[1], new_state[2]), r, t
end

# ╔═╡ b06785ff-c044-4873-87b8-b9d4d977bf88
function Base.show(io::IO, env::GridWithWalls)
	model = fill("□", size(env.walls)...)
	model[env.walls] .= "▤"
	model[env.state[1], env.state[2]] = "◍"
	for row in eachrow(model)
		println(io, join(row, " "))
	end
end

# ╔═╡ 69644819-7549-43ed-ba0e-912e18fdc2c8
@recipe function f(env::GridWithWalls)
    ticks := nothing
    foreground_color_border := nothing
    grid := false
    legend := false
    aspect_ratio := 1
    xaxis := false
    yaxis := false
    yflip := false
	addagent --> true

    SIZE=100
    BG = Colors.RGB(1.0, 1.0, 1.0)
    BORDER = Colors.RGB(0.0, 0.0, 0.0)
    WALL = Colors.RGB(0.3, 0.3, 0.3)
    AC = Colors.RGB(0.69921875, 0.10546875, 0.10546875)
    GOAL = Colors.RGB(0.796875, 0.984375, 0.76953125)

    cell = fill(BG, SIZE, SIZE)
    cell[1, :] .= BORDER
    cell[end, :] .= BORDER
    cell[:, 1] .= BORDER
    cell[:, end] .= BORDER

	wall_cell = fill(WALL, SIZE, SIZE)
    wall_cell[1, :] .= BORDER
    wall_cell[end, :] .= BORDER
    wall_cell[:, 1] .= BORDER
    wall_cell[:, end] .= BORDER


    s_y = size(env.walls, 1)
    s_x = size(env.walls, 2)

    screen = fill(BG, (s_y + 2)*SIZE, (s_x + 2)*SIZE)

    screen[:, 1:SIZE] .= WALL
    screen[1:SIZE, :] .= WALL
    screen[end-(SIZE-1):end, :] .= WALL
    screen[:, end-(SIZE-1):end] .= WALL

    for j ∈ 1:s_x
        for i ∈ 1:s_y
            sqr_i = ((i)*SIZE + 1):((i+1)*SIZE)
            sqr_j = ((j)*SIZE + 1):((j+1)*SIZE)
            if env.state[1] == i && env.state[2] == j
				if plotattributes[:addagent]
	                v = @view screen[sqr_i, sqr_j]
                	v .= cell
                	v[Int(SIZE/2)-Int(SIZE/4):Int(SIZE/2)+Int(SIZE/4)+1, 
					  Int(SIZE/2)-Int(SIZE/4):Int(SIZE/2)+Int(SIZE/4)+1] .= AC
				end
			elseif env.walls[i, j]
                screen[sqr_i, sqr_j] .= wall_cell
			elseif !env.walls[i, j]
                screen[sqr_i, sqr_j] .= cell
            end
        end
    end
    screen[end:-1:1,:]
end

# ╔═╡ 23185969-603e-452b-a3ab-ae624c7b539e
@recipe function f(env::GridWithWalls, bdemon::BDemon)
    ticks := nothing
    foreground_color_border := nothing
    grid := false
    legend := false
    aspect_ratio := 1
    xaxis := false
    yaxis := false
    yflip := false

    SIZE=100

	@series begin
		addagent := false
		env
	end

	s_y = size(env.walls, 1)
    s_x = size(env.walls, 2)

	q_x = Float64[]
	q_y = Float64[]
	q_u = Float64[]
	q_v = Float64[]

	colors = Symbol[]
	
	for j ∈ 1:s_x
        for i ∈ 1:s_y
			if !env.walls[i, j]
				sqr_i = ((s_y - (i-2))*SIZE + 1):-1:((s_y - (i-1))*SIZE)
            	sqr_j = ((j)*SIZE + 1):((j+1)*SIZE)	
				med_sqr_i = median(sqr_i)
				med_sqr_j = median(sqr_j)
				
				q = predict(bdemon, MinimalRLCore.get_state(env, (i, j)))
				actions = findall((x)->x≈maximum(q), q)
				# @info maximum(q)
				for a in actions
					if a == env.UP
						# @info q, med_sqr_j, med_sqr_i
						push!(q_x, med_sqr_j)
						push!(q_y, med_sqr_i + 1)
						push!(q_u, 0)
						push!(q_v, 1*SIZE/2 - 6)
						push!(colors, :orange)
					elseif a == env.DOWN
						push!(q_x, med_sqr_j)
						push!(q_y, med_sqr_i - 1)
						push!(q_u, 0)
						push!(q_v, -1*SIZE/2 + 6)
						push!(colors, :red)
					elseif a == env.RIGHT
						push!(q_x, med_sqr_j + 1)
						push!(q_y, med_sqr_i)
						push!(q_u, 1*SIZE/2 - 6)
						push!(q_v, 0)
						push!(colors, :yellow)
					elseif a == env.LEFT
						push!(q_x, med_sqr_j - 1)
						push!(q_y, med_sqr_i)
						push!(q_u,  -1*SIZE/2 + 6)
						push!(q_v, 0)
						push!(colors, :blue)
					end
				end
			end
		end
	end

	@series begin
		seriestype := :quiver
		arrow := true
		c := :black
		# c := repeat(colors, inner=4)
		# seriescolor := repeat(colors, inner=4)
		# markercolor := repeat(colors, inner=4)
		# linecolor := repeat(colors, inner=4)
		# c := repeat(colors, inner=4)
		# line_z := repeat(colors, inner=4)
		linewidth := 2
		label := ""
		gradient := (q_u, q_v)
		q_x, q_y
	end
	
end

# ╔═╡ 8bf4d4aa-3899-4cf0-b5e1-22db7f1a97a8
gr()

# ╔═╡ f9b6c7ac-f243-46da-9533-5acb7faaa861
let
	env = FourRooms()
	env.state = [1, 11]
	step!(env, env.DOWN)
	plot(env)
end

# ╔═╡ 34e7dfc9-1091-4852-8b38-12d366ec0a05
let
	env = FourRooms()
	env.state = [1, 11]
	step!(env, env.DOWN)
	env
end

# ╔═╡ 60648ab1-27f8-4fbc-abd6-d26f25522151


# ╔═╡ e2c6be01-f399-4acf-8b9a-4a89b9c3e6a3
# function which_room(env::GridWithWalls, state)
#     frp = FourRoomsParams
#     room = -1
#     if state[1] < 6
#         # LEFT
#         if state[2] < 6
#             # TOP
#             room = frp.ROOM_TOP_LEFT
#         else
#             # Bottom
#             room = frp.ROOM_BOTTOM_LEFT
#         end
#     else
#         # RIGHT
#         if state[2] < 7
#             # TOP
#             room = frp.ROOM_TOP_RIGHT
#         else
#             # Bottom
#             room = frp.ROOM_BOTTOM_RIGHT
#         end
#     end
#     return room
# end

# ╔═╡ 1c793c5e-a2d6-4160-b672-281017b6aeae
md"""
# Cycle World Experiments

- GVF compositional chain. With first GVF as a prediction with c=1 at the head of the cycle.
- Tabular feature representation.
- Each GVF after the first is normalized by (1-\gamma)

"""

# ╔═╡ 6f50c308-4f63-42f1-a80c-0bbbe09f5632
md"""
## Experiment Function
"""

# ╔═╡ 47390cc3-cf50-4647-8ba9-76539f0e7916
function cycleworld_experiment!(
		horde,
		env_size, 
		num_steps, 
		lu = TDλ(0.1, 0.9); kwargs...)

	
	env = CycleWorld(env_size, partially_observable=false)
	
	s_t = start!(env)
	
	@progress for step in 1:num_steps
		
		s_tp1, r_tp1, _ = step!(env, 1)
		p_tp1 = predict(horde, s_tp1)
		update!(lu, horde, s_t, s_t, nothing, nothing, s_tp1, s_tp1, r_tp1, p_tp1)

		s_t = copy(s_tp1)
	end
	
	p = [predict(horde, [x]) for x in [1:env_size; 1:env_size]]
	
	p
end

# ╔═╡ b8bdcfa7-84f9-4547-997c-a7cc3d6c0521
function cycleworld_experiment(horde_init::Function, args...; kwargs...)
	horde = horde_init()
	p = cycleworld_experiment!(horde, args...; kwargs...)
	horde, p
end

# ╔═╡ 0b3f6284-63db-481a-9967-6bc538b677bc
function cw_run_and_plot(horde_init, args...)
	horde, p = cycleworld_experiment(horde_init, args...)
	plot([plot(getindex.(p, i), legend=nothing) for i in 1:length(horde)]...)
end

# ╔═╡ 02e5353b-1536-454b-8526-fb3fd4449a89
md"# Myopic Chain"

# ╔═╡ 5019fe8f-8755-4f61-8f7a-e8fa14e39fa1
# cw_myopic_hrd, cw_myopic_p = let	
# 	env_size = 10
# 	num_steps = 100000
# 	lu = TDλ(0.1, 0.9)
# 	γ = 0.0
# 	horde, p = cycleworld_experiment(env_size, num_steps, TDλ(0.1, 0.9)) do 
# 		[[GVF(env_size, 
# 			FeatureCumulant(1), 
# 			OnPolicy(), 
# 			ConstantDiscount(γ))];
# 		[GVF(env_size, 
# 			PredictionCumulant(i), 
# 			OnPolicy(), 
# 			ConstantDiscount(γ)) for i in 1:9]]
# 	end
# 	horde, p
# end

# ╔═╡ 7d5013b6-c0a7-452a-8e06-2ea5c21c1788
let
	# plotly()
	horde = cw_myopic_hrd
	num_iterations = 4
	p = [predict(horde, [x]) for x in [reduce(vcat, fill(1:10, num_iterations)); [1, 2]]]
	x_top = 10*num_iterations
	
	cw_obs = fill(0, 10*num_iterations + 1)
	for i in 1:(x_top+1)
		if i % 10 == 1
			cw_obs[i] = 1
		end
	end
	plt = bar(
		1:x_top+1, 
		cw_obs, 
		bar_width=0.01, 
		xrange=(0, x_top+2.1), 
		yrange=(0.0, 1.1), 
		legend=false, 
		grid=false, 
		tick_dir=:out,
		yformatter=(a)->"", xformatter=(a)->"", 
		yticks=false)
	
	scatter!(1:x_top+1, cw_obs, color=color_scheme[2])
	
	p_plot(i; kwargs...) = begin
		plot(getindex.(p, i), 
			legend=nothing, 
			xrange=(0, x_top+2.1), 
			lw=2, 
			grid=false, 
			tick_dir=:out, 
			yformatter=(a)->"",
			yticks=false, 
			color=color_scheme[1]; kwargs...)
	end
	plts = [plt]
	plts = [plts; [p_plot(i, xformatter=(a)->"") for i in 1:length(horde)-1]]
	
	plt_end = p_plot(length(horde), xtickfontsize=15)
	plt = plot(plts[1], plts[2], plts[3], plts[4], plt_end, layout = (5, 1))
	plt
end

# ╔═╡ e95c6f92-86bb-4f93-8ed9-10025e0dbf9f
md"# Four Rooms Experiment"

# ╔═╡ 2faab55b-74b0-4e6f-b36b-a85eceb17c87
let
	env = FourRooms()
	env.state = [1, 11]
	env
	MinimalRLCore.get_state(env)
end

# ╔═╡ ab8035d2-3d55-4d82-b078-65a866bdebe0
function fourrooms_experiment!(
		horde, 
		num_steps,
		lu = TDλ(0.1, 0.9); seed = 1, kwargs...)

	rng = Random.Xoshiro(seed)
	
	env = FourRooms() #CycleWorld(env_size, partially_observable=false)
	
	s_t = start!(env, rng)
	
	@progress for step in 1:num_steps

		a_t = rand(rng, env.ACTIONS)
		s_tp1, r_tp1, _ = step!(env, a_t)
		p_tp1 = predict(horde, s_tp1)
		update!(lu, horde, s_t, s_t, a_t, 
			1/length(env.ACTIONS), s_tp1, s_tp1, 
			r_tp1, p_tp1)

		s_t = copy(s_tp1)
	end
	env_size = size(FourRooms())
	env_feat_size = env_size[1] * env_size[2] 
	p = [predict(horde, x) for x in 1:env_feat_size]
	
	p
end

# ╔═╡ a4cb5611-2572-47da-8a74-ad475c0e222e
function fourrooms_experiment(horde_init::Function, args...; kwargs...)
	horde = horde_init()
	p = fourrooms_experiment!(horde, args...; kwargs...)
	horde, p
end

# ╔═╡ 82a4c747-5f01-49d9-aad7-0cac78cc1784
function fourrooms_behavior!(
		bdemon, 
		num_steps,
		lu; seed = 1, kwargs...)

	rng = Random.Xoshiro(seed)
	
	env = FourRooms() #CycleWorld(env_size, partially_observable=false)

	total_steps = 0

	while total_steps < num_steps
		s_t = start!(env, rng)
		start!(bdemon, s_t)
		while is_terminal(bdemon, s_t, s_t) == false
			a_t = get_action(rng, bdemon, s_t)

			s_tp1, r_tp1, _ = MinimalRLCore.step!(env, a_t)
			p_tp1 = predict(bdemon, s_tp1)
			update!(lu, bdemon, s_t, s_t, a_t, nothing, s_tp1, s_tp1, r_tp1, p_tp1)

			total_steps += 1
			s_t = copy(s_tp1)
			total_steps < num_steps || break
		end
		# bdemon.z .= 0
	end
	env_size = size(FourRooms())
	env_feat_size = env_size[1] * env_size[2] 
	p = [predict(bdemon, x) for x in 1:env_feat_size]
	
	p
end

# ╔═╡ c8b527c4-cdd7-4b42-9cec-5ab45b25fbc1
function fourrooms_behavior_offpolicy!(
		bdemon, 
		num_steps,
		lu; kwargs...)

	
	env = FourRooms() #CycleWorld(env_size, partially_observable=false)

	total_steps = 0
	while total_steps < num_steps
		s_t = start!(env, Random.default_rng())
		# while is_terminal(bdemon, s_t, s_t) == false
			# a_t = get_action(bdemon, s_t)
			a_t = rand(1:4)

			s_tp1, r_tp1, _ = step!(env, a_t)
			p_tp1 = predict(bdemon, s_tp1)
			update!(lu, bdemon, s_t, s_t, a_t, nothing, s_tp1, s_tp1, r_tp1, p_tp1)

			total_steps += 1
			s_t = copy(s_tp1)
			
			# (is_terminal(bdemon, s_tp1, s_tp1) == false) || break
			# total_steps < num_steps || break
		# end
		# bdemon.z .= 0
	end
	@info total_steps
	env_size = size(FourRooms())
	env_feat_size = env_size[1] * env_size[2] 
	p = [predict(bdemon, x) for x in 1:env_feat_size]
	
	p
end

# ╔═╡ 495b205a-8bc9-4c7a-943c-1921362cd296
function fourrooms_heatmap_valuefunction(p::AbstractVector)
	heatmap(reshape(p, 11, 11)[:, end:-1:1]')
end

# ╔═╡ 94e12aa5-fcd8-4656-a7c0-3c09f1c6dbd1
function fourrooms_plot_policy(policy::AbstractVector)
	# heatmap(reshape(p, 11, 11)[:, end:-1:1]')
	
end

# ╔═╡ 1746c4a8-7b2b-4187-825f-be978af4ace9
fr_gamma_hrd, fr_gamma_p = let	
	num_steps = 3_000_000
	lu = TDλ(0.1, 0.9)
	γ = 0.9
	env_size = size(FourRooms())
	env_feat_size = env_size[1] * env_size[2] 
	horde, p = fourrooms_experiment(num_steps, TDλ(0.1, 0.9)) do 
		[[GVF(env_feat_size, 
			FeatureCumulant(11), 
			OnPolicy(), 
			ConstantDiscount(γ))
			# TerminatingDiscount(γ, 11))
		];
		[GVF(env_feat_size, 
			RescaleCumulant(
				PredictionCumulant(i), 
				γ),
			OnPolicy(), 
			ConstantDiscount(γ))
			# TerminatingDiscount(γ, 11)) 
			for i in 1:20]]
	end
	horde, p
end

# ╔═╡ 5cfb16ca-463c-4119-8369-0c9742a3a8eb
fourrooms_heatmap_valuefunction(getindex.(fr_gamma_p, 10))

# ╔═╡ 44ae7ccf-fdd2-4f8a-b7c7-543d649d57fc
fourrooms_heatmap_valuefunction(getindex.(fr_gamma_p, 20))

# ╔═╡ 74a4a631-514a-4f74-82ff-ab68ddab4977
fr_term_hrd, fr_term_p = let
	num_steps = 1_000_000
	lu = TDλ(0.01, 0.9)
	γ = 0.9
	env_size = size(FourRooms())
	env_feat_size = env_size[1] * env_size[2] 
	horde, p = fourrooms_experiment(num_steps, TDλ(0.01, 0.9)) do 
		[[GVF(env_feat_size, 
			FeatureCumulant(11), 
			OnPolicy(), 
			# ConstantDiscount(γ))
			TerminatingDiscount(γ, 11))
		];
		[GVF(env_feat_size, 
			PredictionCumulant(i), 
			OnPolicy(), 
			# ConstantDiscount(γ))
			TerminatingDiscount(γ, 11)) 
			for i in 1:11]]
	end
	horde, p
end

# ╔═╡ fcf6606f-8c16-4c27-8813-419cb53bf0e1
fourrooms_heatmap_valuefunction(getindex.(fr_term_p, 1))

# ╔═╡ 58f69245-65e8-4e8e-a63d-c2000e77fbd4
fourrooms_heatmap_valuefunction(getindex.(fr_term_p, 2))

# ╔═╡ 8fa48e9f-ef7e-4d99-a391-3b4268bd2ebe
md"## Behavior Demon"

# ╔═╡ fe0c2f4e-2ed9-4c62-928b-a617bd1a5e36
fr_bdeomon_11_pred, fr_bdemon_11 = let
	env_size = size(FourRooms())
	env_feat_size = env_size[1] * env_size[2] 
	bdemon = BDemon(env_feat_size, 4, FeatureCumulant(11), ϵGreedy(0.1),
	TerminatingDiscount(0.9, 11))
	fourrooms_behavior!(bdemon, 1_000_000, Qλ(0.1, 0.9)), bdemon
end

# ╔═╡ fb0edc00-0d2f-478c-ab8d-7d61f399b465
plot(FourRooms(), fr_bdemon_11)

# ╔═╡ a2634068-2f4d-4986-9d2d-f3e8bf729c4c
plot(FourRooms(), fr_bdemon_11)

# ╔═╡ cc34f26e-b9d9-479b-a962-c98823244ebb
fourrooms_heatmap_valuefunction(getindex.(fr_bdeomon_11_pred, 1))

# ╔═╡ 8ae1ff0e-d129-4496-ae0e-e10ef0ec7d31
fr_bd_hrd, fr_bd_p = let	
	num_steps = 2_000_000
	lu = TDλ(0.01, 0.9)
	γ = 0.9
	env_size = size(FourRooms())
	env_feat_size = env_size[1] * env_size[2] 
	horde, p = fourrooms_experiment(num_steps, lu) do 
		[[GVF(env_feat_size, 
			FeatureCumulant(11),
			fr_bdemon_11, 
			# ConstantDiscount(γ))
			TerminatingDiscount(γ, 11))
		];
		[GVF(env_feat_size, 
			RescaleCumulant(
				PredictionCumulant(i), 
				γ),
			fr_bdemon_11, 
			# ConstantDiscount(γ))
			TerminatingDiscount(γ, 11))
			for i in 1:11]]
	end
	horde, p
end

# ╔═╡ 8665d2ad-e5bc-417e-b347-8a01fa90ce93
fourrooms_heatmap_valuefunction(getindex.(fr_bd_p, 4))

# ╔═╡ 9d4bc54c-2c4a-463c-86c5-bf43ecc2aafc
fr_comp_bdemon_p, fr_comp_bdemon = let	
	env_size = size(FourRooms())
	env_feat_size = env_size[1] * env_size[2] 
	bdemon = BDemon(
		env_feat_size, 
		4, 
		GVFCumulant(fr_bd_hrd[4]),
		# FeatureCumulant(11), 
		ϵGreedy(0.1),
		GVFThreshTerminatingDiscount(0.9, fr_bd_hrd[4], 0.23))
	
	fourrooms_behavior_offpolicy!(bdemon, 500_000, Qλ(0.01, 0.9)), bdemon
end

# ╔═╡ ff876ea8-dc3e-4479-a246-65d16b5b85e2
fourrooms_heatmap_valuefunction(getindex.(fr_comp_bdemon_p, 1))

# ╔═╡ 8763f124-96b4-47de-bd33-bfd7b391e056
plot(FourRooms(), fr_comp_bdemon)

# ╔═╡ df9a9c7a-24ad-449c-9744-8e13d31e94c5
fr_comp_bdemon_max_p, fr_comp_bdemon_max = let	
	env_size = size(FourRooms())
	env_feat_size = env_size[1] * env_size[2] 
	bdemon = BDemon(
		env_feat_size, 
		4, 
		GVFCumulant(fr_bd_hrd[4]),
		# FeatureCumulant(11), 
		ϵGreedy(0.1),
		GVFThreshTerminatingMaxDiscount(0.9, fr_bd_hrd[4]))
	
	fourrooms_behavior_offpolicy!(bdemon, 500_000, Qλ(0.01, 0.9)), bdemon
end

# ╔═╡ 677e1b65-0a5b-4d40-b083-7a3f84692672
fourrooms_heatmap_valuefunction(getindex.(fr_comp_bdemon_max_p, 4))

# ╔═╡ 53c46316-3fa9-4e52-be60-862d58e31081
fr_comp_bdemon_11_p, fr_comp_bdemon_11 = let	
	env_size = size(FourRooms())
	env_feat_size = env_size[1] * env_size[2] 
	bdemon = BDemon(
		env_feat_size, 
		4, 
		GVFCumulant(fr_bd_hrd[11]),
		# FeatureCumulant(11), 
		ϵGreedy(0.1),
		GVFThreshTerminatingDiscount(0.9, fr_bd_hrd[4], 0.225))
	
	fourrooms_behavior_offpolicy!(bdemon, 200_000, Qλ(0.01, 0.9)), bdemon
end

# ╔═╡ 89bb8312-734b-4800-8bfd-93fbee37052b
fourrooms_heatmap_valuefunction(getindex.(fr_comp_bdemon_11_p, 1))

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
MinimalRLCore = "4557a151-568a-41c4-844f-9d8069264cea"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
Colors = "~0.12.10"
MinimalRLCore = "~0.2.1"
Plots = "~1.38.6"
PlutoUI = "~0.7.50"
ProgressLogging = "~0.1.4"
RecipesBase = "~1.3.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "6f7f4a5fefb88827acee40faa1d93754677fb371"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c6d890a52d2c4d55d326439580c3b8d0875a77d9"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.7"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "485193efd2176b88e6622a39a246f8c5b600e74e"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.6"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random", "SnoopPrecompile"]
git-tree-sha1 = "aa3edc8f8dea6cbfa176ee12f7c2fc82f0608ed3"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.20.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonRLInterface]]
deps = ["MacroTools"]
git-tree-sha1 = "21de56ebf28c262651e682f7fe614d44623dc087"
uuid = "d842c3ba-07a1-494f-bbec-f5741b0a3e98"
version = "0.3.1"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "61fdd77467a5c3ad071ef8277ac6bd6af7dd4c04"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "660b2ea2ec2b010bb02823c6d0ff6afd9bdc5c16"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.71.7"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "d5e1fd17ac7f3aa4c5287a61ee28d4f8b8e98873"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.71.7+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "37e4657cd56b11abe3d10cd4a1ec5fbdb4180263"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.7.4"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "2422f47b34d4b127720a18f86fa7b1aa2e141f29"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.18"

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "0a1b7c2863e44523180fdb3146534e265a91870b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.23"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MinimalRLCore]]
deps = ["CommonRLInterface", "Lazy", "LinearAlgebra", "Random", "StatsBase"]
git-tree-sha1 = "0e13b3968a1247f5578d104f44c74545678c108a"
uuid = "4557a151-568a-41c4-844f-9d8069264cea"
version = "0.2.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "6503b77492fd7fcb9379bf73cd31035670e3c509"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.3.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9ff31d101d987eb9d66bd8b176ac7c277beccd09"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.20+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.40.0+0"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "6f4fbcd1ad45905a5dee3f4256fabb49aa2110c6"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.7"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "c95373e73290cf50a8a22c3375e4625ded5c5280"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "da1d3fb7183e38603fcdd2061c47979d91202c97"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.6"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "5bb5129fdd62a2bbbe17c2756932259acf467386"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.50"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "261dddd3b862bd2c940cf6ca4d1c8fe593e457c8"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.3"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase", "SnoopPrecompile"]
git-tree-sha1 = "e974477be88cb5e3040009f3767611bc6357846f"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.11"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "94f38103c984f89cf77c402f2a68dbd870f8165f"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.11"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c6edfe154ad7b313c01aceca188c05c835c67360"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.4+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╠═7bc2d377-7c94-4cee-8e3a-e8ce50a471b3
# ╠═f6a92f82-be00-11ed-14fc-450478eca211
# ╠═fd86d35c-9433-4e68-88b8-078c56bcf2a4
# ╠═94b20a16-cccf-4512-967d-77261222544b
# ╠═2bf20b76-74e2-42b2-8053-fa66657c5563
# ╠═3b85f9cd-f9aa-47ac-852d-ed7839167790
# ╠═e09f74b0-fff2-4417-8867-f05f47c6a055
# ╟─a155572b-9273-491d-bcbe-9540efd09081
# ╟─4dab38f3-a07f-46e5-8bcd-26132c65ca43
# ╟─5ab59373-6a80-46b2-8687-ade15aa31b5e
# ╟─c9f0f1e2-d596-45d4-9a6b-ead881e615f3
# ╠═440a6393-d3f2-4158-9cad-a9a3ffb658b7
# ╠═49c35c5b-e99c-4768-99fd-6344f8ebd3e1
# ╠═e3a469fb-98d1-4a17-b1f4-2dc77a8224a4
# ╠═3f3048ff-843f-41b0-aa86-b64090cddc57
# ╠═a22330b9-c3f7-42da-929b-1023ccea6df2
# ╠═ca873653-1b35-42a7-a3f6-b0b6f8c3d314
# ╠═0c583b98-3740-4812-9969-e2d00d03ce3d
# ╠═8abd1d2e-6519-4352-9859-4ed1f3bfc1d0
# ╠═184ecfc1-127f-4585-a618-3d159f79d718
# ╠═3527021b-de33-42da-aac0-619483caba3e
# ╠═713bdb3f-7afe-4215-9191-b7704383c6b0
# ╠═73ae37d9-2a85-4e3b-b2b0-29b9b0cbc029
# ╠═5e6ded46-1ff1-4667-966c-7e2bd4a35945
# ╠═349185f3-855d-40b7-a669-8468e0bb686c
# ╠═b8130b16-52c9-42e2-8e87-03e047547809
# ╠═2756de72-2851-43cd-a058-da96d294b70c
# ╠═0939ea40-1170-4269-9e6e-6ba5a0c80d72
# ╠═b1bc29d3-9790-47fc-9ca5-f8215ffa1188
# ╠═dfb951c9-5bfa-4307-864f-af02c5c1578d
# ╠═6d1f2c5c-90e5-41e7-a6b9-1abb7f2426ad
# ╠═2e621539-afc9-4306-8f2b-d418099f5a34
# ╠═7bbe957e-0669-42e5-9384-e31fc31f8283
# ╠═c2a35607-bf5c-441f-a708-53f32101f33e
# ╠═51e69bed-bf48-4688-87fd-d0c9e9bfc34e
# ╠═fa854b1a-a28c-4a28-b8e4-3e03a4a99810
# ╠═d135f4fe-8700-4a60-acec-890a9dc562fc
# ╠═26a2fd60-0812-404b-a1d5-145a24d6fe35
# ╟─6481cbd3-7ae0-4adc-b7bd-d4fa1957defa
# ╠═d6e1b5c1-568d-43d5-a90f-dd0dd3c907f7
# ╠═0ba53300-2ff7-4c59-b7c9-d9896ed497af
# ╠═4e01a022-5bb6-47de-81d3-9ff418198f6c
# ╠═8a44f9bd-b7ab-4102-8234-d73f7e979fa9
# ╠═594b5d03-9dc9-4df7-bef4-a71a81224e27
# ╟─b8e3c48e-e7f6-402f-8f44-08663bd273a2
# ╠═b4c75f1b-fbc5-4370-8241-f51e7865fcdd
# ╠═7b3b895a-4563-42be-8d4f-5cb9dd53a6d7
# ╠═32c22f6c-02b2-4064-9b79-eeed1bb975ce
# ╟─0398417f-db77-4ec7-b385-9f3206a5365c
# ╟─47eba562-99e0-4815-a028-21f7376cd257
# ╠═44bba6f3-5f82-4c9a-b4da-e33f23010ee3
# ╠═a69fed13-737d-4fd6-8baf-0dabe7f339a5
# ╠═356c80aa-d825-45dc-8708-cf7ad6ede14f
# ╠═217a9a10-1b57-4d4a-8b41-eeaea71c749f
# ╠═5001c4a2-343d-4f5e-a2e4-4e01750932e9
# ╠═695fd511-5f59-4e77-8b26-5e3787dbc198
# ╠═708a85fd-9b52-420b-a468-f8b4129ce794
# ╠═f7405d98-10e9-4e2e-ac13-1780dc6c5446
# ╠═ba9085c6-cecb-4ea5-8f3d-0ed21a23c196
# ╠═cda934f1-08f7-460a-b4b2-c9c20ade52a4
# ╠═3cff48e4-ad25-4354-8cdb-877586b33e96
# ╠═660e8864-012f-4347-abf0-777d8c20aecd
# ╠═1b9ec5be-ca0f-4221-abf1-17f44b5ba118
# ╠═6733a1bb-88bd-4e01-9a03-dd39279ba1e7
# ╠═33db32af-2e34-4dad-abcc-e023349aae70
# ╠═3bffcae1-1570-4d9f-814a-702afd51d3e4
# ╠═204234d7-212a-4985-a45d-1779286760e1
# ╠═a351143c-a221-4c44-94a6-249308c38a11
# ╠═55a73859-6da8-4d4f-a391-a9bf63a2f98c
# ╟─0bc9f5c5-6fb4-4874-8820-3345f50055e6
# ╟─dd006581-f89c-426c-9ade-7f7edb58dc88
# ╠═981a4a12-8f59-4d8a-b734-1993280959e6
# ╠═d864d151-5d71-4851-a8ec-72e415573476
# ╠═c0c9cf24-7294-4e26-8d0f-f978c9996e04
# ╠═e52b2a09-e4f7-4756-96a8-6dce21a3eaac
# ╠═4bffba42-2d29-46ac-9bdd-59f348d92a30
# ╠═31115f9d-c987-4b11-9b02-ee7942606f7a
# ╠═7b266d56-2a4d-40de-9a9f-3a9680bebacc
# ╠═a2abdbd6-c2a4-443f-b724-6bf574436ee7
# ╠═924b6945-3826-4138-aa00-206ddb9840a5
# ╠═f18749cd-0160-44ed-b91f-30a6a9dfad8f
# ╠═1775dc2a-91ea-4303-a5c7-b260dd06278d
# ╟─940461dc-735f-4297-ad5d-f67a23ebbb5f
# ╠═1e62ec00-975b-492c-bb11-f7b7892a58e8
# ╠═70d5188f-20dd-49cd-a18d-aeefa16d8bb2
# ╠═621e43ea-2006-472e-b6a5-94ae68d07bf2
# ╠═ab3597f3-ed93-42be-b22a-74c9a507cc57
# ╠═9b9bc9ae-5ea5-4f1d-afaf-7d4d20b2baf5
# ╠═f48d5184-2b3c-4fea-9948-66332e40d74d
# ╠═6bc7973a-98a2-40dd-8a2d-aa27887c8fa1
# ╠═841a7a9f-fc8d-4107-a411-3039e1826325
# ╠═bdb1b1da-f2d3-4248-b5bd-d6135c61d89c
# ╠═b16223bf-e803-4c08-9d68-e688db7351e5
# ╠═ab1bf706-9b9c-4b2f-b68b-65dfd6acf04b
# ╠═07501680-edfd-4d17-b399-d37bb7806ee0
# ╠═6f194e0d-e844-4d4c-b946-7ea8b75327fc
# ╠═992f3a2d-a942-4d72-8f7d-c37f904e915f
# ╠═983c6296-db06-419c-81ed-eac04fa4db75
# ╠═b06785ff-c044-4873-87b8-b9d4d977bf88
# ╠═69644819-7549-43ed-ba0e-912e18fdc2c8
# ╠═23185969-603e-452b-a3ab-ae624c7b539e
# ╠═8bf4d4aa-3899-4cf0-b5e1-22db7f1a97a8
# ╠═fb0edc00-0d2f-478c-ab8d-7d61f399b465
# ╠═f9b6c7ac-f243-46da-9533-5acb7faaa861
# ╠═34e7dfc9-1091-4852-8b38-12d366ec0a05
# ╠═60648ab1-27f8-4fbc-abd6-d26f25522151
# ╠═e2c6be01-f399-4acf-8b9a-4a89b9c3e6a3
# ╟─1c793c5e-a2d6-4160-b672-281017b6aeae
# ╠═6f50c308-4f63-42f1-a80c-0bbbe09f5632
# ╠═b8bdcfa7-84f9-4547-997c-a7cc3d6c0521
# ╠═47390cc3-cf50-4647-8ba9-76539f0e7916
# ╠═0b3f6284-63db-481a-9967-6bc538b677bc
# ╟─02e5353b-1536-454b-8526-fb3fd4449a89
# ╠═5019fe8f-8755-4f61-8f7a-e8fa14e39fa1
# ╟─7d5013b6-c0a7-452a-8e06-2ea5c21c1788
# ╟─e95c6f92-86bb-4f93-8ed9-10025e0dbf9f
# ╠═2faab55b-74b0-4e6f-b36b-a85eceb17c87
# ╠═a4cb5611-2572-47da-8a74-ad475c0e222e
# ╠═ab8035d2-3d55-4d82-b078-65a866bdebe0
# ╠═82a4c747-5f01-49d9-aad7-0cac78cc1784
# ╠═c8b527c4-cdd7-4b42-9cec-5ab45b25fbc1
# ╠═495b205a-8bc9-4c7a-943c-1921362cd296
# ╠═94e12aa5-fcd8-4656-a7c0-3c09f1c6dbd1
# ╠═1746c4a8-7b2b-4187-825f-be978af4ace9
# ╠═5cfb16ca-463c-4119-8369-0c9742a3a8eb
# ╠═44ae7ccf-fdd2-4f8a-b7c7-543d649d57fc
# ╠═74a4a631-514a-4f74-82ff-ab68ddab4977
# ╠═fcf6606f-8c16-4c27-8813-419cb53bf0e1
# ╠═58f69245-65e8-4e8e-a63d-c2000e77fbd4
# ╠═8fa48e9f-ef7e-4d99-a391-3b4268bd2ebe
# ╠═fe0c2f4e-2ed9-4c62-928b-a617bd1a5e36
# ╠═a2634068-2f4d-4986-9d2d-f3e8bf729c4c
# ╠═cc34f26e-b9d9-479b-a962-c98823244ebb
# ╠═8ae1ff0e-d129-4496-ae0e-e10ef0ec7d31
# ╠═8665d2ad-e5bc-417e-b347-8a01fa90ce93
# ╠═9d4bc54c-2c4a-463c-86c5-bf43ecc2aafc
# ╠═ff876ea8-dc3e-4479-a246-65d16b5b85e2
# ╠═8763f124-96b4-47de-bd33-bfd7b391e056
# ╠═df9a9c7a-24ad-449c-9744-8e13d31e94c5
# ╠═677e1b65-0a5b-4d40-b083-7a3f84692672
# ╠═53c46316-3fa9-4e52-be60-862d58e31081
# ╠═89bb8312-734b-4800-8bfd-93fbee37052b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
