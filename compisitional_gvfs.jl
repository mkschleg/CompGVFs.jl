### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ ebe424a8-f385-46ee-80e9-12b99bb2e691
using Random

# ╔═╡ 098f4309-d067-4d4e-be6c-8dacdc059b89
using PlutoUI, StatsPlots, Plots

# ╔═╡ b4d08bb8-ea59-11eb-1df0-93d1149fcf0a
using LoopVectorization, Tullio, MinimalRLCore

# ╔═╡ 917e3c4f-75c7-4759-81d5-a0c6823d784a
using DSP

# ╔═╡ 729faad8-176f-4b6b-af97-3e321c7be3e0
md"""
- [Time As a Variable: Time-Series Analysis](https://www.oreilly.com/library/view/data-analysis-with/9781449389802/ch04.html)
- [A Very Short Course on Time Series Analysis](https://bookdown.org/rdpeng/timeseriesbook/filtering-time-series.html)
"""

# ╔═╡ 5590501e-eab5-4d08-a81f-b1f5a2b91a08
import RDatasets, DataFrames, PyCall

# ╔═╡ 1df5d910-1a95-4049-906f-770eb3a7990a
import FourierAnalysis

# ╔═╡ 84c1f22f-5a42-45de-86f0-3729922336e1
# Setup Plots
let
	plotly()
	plot(rand(10), size=(5,5))
end

# ╔═╡ 64848d48-09a0-4e13-a318-877b66a4f5e6
PlutoUI.TableOfContents(title="Comp-GVFS", indent=true, aside=true)

# ╔═╡ 4f5da656-f619-416f-9421-0a21fca291af
md"""
# GVF details
"""

# ╔═╡ cefce664-1c9e-435e-b6bc-1be062d95e1b
md"""
## Parameter Functions
"""

# ╔═╡ 76453841-7bec-4fd4-bdd1-1ea185f26e13
md"""
### Cumulants
"""

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

# ╔═╡ 2b14f50c-3b46-42ad-a01c-5e47d31914e4
md"""
### Policies
"""

# ╔═╡ 131b2d9b-711b-4cea-bab1-03f0ef68f5a9
struct OnPolicy end

# ╔═╡ 91fdbc6f-479c-4a79-848b-b0a83268348b
get_value(op::OnPolicy, args...) = 1.0

# ╔═╡ 56f4136d-3e82-47ac-91a3-48f7331ef7c7
md"""
### Discount
"""

# ╔═╡ 725c9586-615d-4d27-8a2f-fe2760aeaedc
struct ConstantDiscount{F}
	γ::F
end

# ╔═╡ 7cbb8f85-1fd3-4c0d-a081-0d9c487227e6
get_value(cd::ConstantDiscount, args...) = cd.γ

# ╔═╡ 4d7bcae2-e7dd-4aa4-84e5-6c529be7c2b4
struct TerminatingDiscount{F}
	γ::F
	idx::Int
end

# ╔═╡ 85392583-6481-4a77-96c0-30f136e08299
get_value(fc::TerminatingDiscount, o, x::Vector{Int}) = fc.idx ∈ x ? zero(typeof(fc.γ)) : fc.γ

# ╔═╡ 2042308c-a555-4ab2-8eec-75925659b504
md"""
## GVF
"""

# ╔═╡ 2410a3a2-f1d6-4edf-ae52-66d43816093b
# What are we doing.
# Massively parallel linear GVF learning
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
predict(gvf::GVF{<:AbstractVector}, x::AbstractVector{<:Number}) = 
	dot(gvf.w, x)

# ╔═╡ b40e514c-04d0-4690-ad89-b81761634bf4
predict(gvf::GVF{<:AbstractVector}, x::AbstractVector{Int}) = begin; 
	w = gvf.w; 
	@tullio ret := w[x[i]]; 
end

# ╔═╡ 2b3f8086-9f7d-4fa7-bfb9-a9a5746fcd5d
predict(gvf::GVF{<:AbstractVector}, x::Int) = begin; 
	gvf.w[x]
end

# ╔═╡ 010f3a6c-e331-4609-b180-638914171c18
md"""
## Horde
"""

# ╔═╡ 0a9dbac5-195b-4c78-a0e4-62f6727e4fce
const Horde = Vector{<:GVF}

# ╔═╡ 0ce6eaec-1c2f-478b-96fd-fe4965517e46
predict(horde::Horde, x) = [predict(gvf, x) for gvf in horde]

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

# ╔═╡ e951e11c-fef2-4bd6-85dd-0c5186bf1771
md"""
# Environments
"""

# ╔═╡ 8126a898-1d98-4013-91a6-8acba476556a
begin
	
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

	function MinimalRLCore.reset!(env::CycleWorld, rng=nothing; kwargs...)
		env.agent_state = 0
	end

	MinimalRLCore.get_actions(env::CycleWorld) = Set(1:1)

	function MinimalRLCore.environment_step!(
			env::CycleWorld,
			action, 
			rng=nothing; 
			kwargs...)
		env.agent_state = (env.agent_state + 1) % env.chain_length
	end


	MinimalRLCore.get_reward(env::CycleWorld) = 0 # -> get the reward of the environment

	function MinimalRLCore.get_state(env::CycleWorld) # -> get state of agent
		if env.partially_observable
			return partially_observable_state(env)
		else
			return fully_observable_state(env)
		end
	end

	fully_observable_state(env::CycleWorld) = [env.agent_state+1]

	function partially_observable_state(env::CycleWorld)
		state = zeros(1)
		if env.agent_state == 0
			state[1] = 1
		end
		return state
	end

	function MinimalRLCore.is_terminal(env::CycleWorld) # -> determines if the agent_state is terminal
		return false
	end

	function Base.show(io::IO, env::CycleWorld)
		model = fill("0", env.chain_length)
		model[1] = "1"
		println(io, join(model, ' '))
		model = fill("-", env.chain_length)
		model[env.agent_state + 1] = "^"
		println(io, join(model, ' '))
		# println(env.agent_state)
	end
	
	md"""
	## Define Cycle World
	
	press eyeball to inspect code.
	"""
end

# ╔═╡ 81f79509-8f08-4df3-8074-2711e6fbdca7
begin
	mutable struct MSO <: MinimalRLCore.AbstractEnvironment
		θ::Int
		Ω::Vector{Float64}
		state::Vector{Float64}
	end

	MSO() = MSO(1, [0.2, 0.311, 0.42, 0.51], [0.0])

	function MinimalRLCore.start!(self::MSO)
		self.θ = 1
		return step!(self)
	end

	function MinimalRLCore.step!(self::MSO)
		self.state[1] = sum([sin(self.θ*ω) for ω in self.Ω])
		self.θ += 1
		return self.state
	end

	MinimalRLCore.get_state(self::MSO) = self.state

	get_num_features(self::MSO) = 1
	get_num_targets(self::MSO) = 1
	
	md"""
	## Define MSO
	"""
end

# ╔═╡ 1568abbd-856f-46bb-bf30-86ee3e6553c6
md"""
# Data Transforms
"""

# ╔═╡ 91957826-379f-4db4-9a61-fc7c5fa667b1
md"""
## Monte Carlo
"""

# ╔═╡ 931db888-1f65-4ac8-965a-e3fa12672ea4
function montecarlo_transform(γ, seq)
	# ret = zeros(eltype(seq), length(seq))
	s = zero(eltype(seq))
	for r in Iterators.reverse(seq)
		s = r + γ*s
	end
	s
end

# ╔═╡ bc620363-6bf2-4c06-b066-f68cbea8a290
function montecarlo_transform_iterate(γ, seq)
	[montecarlo_transform(γ, @view seq[i:end]) for i in 1:length(seq)]
end

# ╔═╡ 611e00d5-cf92-4e0b-9504-8b611b88a897
md"""
## Butterworth Filters
"""

# ╔═╡ 8850c353-20ba-4181-ac5a-70ca86e15066
function butterworth_filter(n, seq; cutoff_freq=0.2)
	zpk = digitalfilter(Lowpass(cutoff_freq), DSP.Butterworth(n))
	sos = convert(SecondOrderSections, zpk)
	DSP.filt(sos, seq)
end

# ╔═╡ 4a1fe221-fb84-4aa4-9691-3dc2fa3893c3
md"""
## Exponential Smoothing

"""

# ╔═╡ 8f094be4-9228-49dd-9b77-89917e4825cf
function exp_smoothing_iterate(exp_smo::Function, seq)
	[exp_smo(@view seq[1:i]) for i in 1:length(seq)]
end

# ╔═╡ c0da6083-792f-4d1f-99c4-6a69c09d01d2
function exp_smoothing(α, X)
	s = X[1]
	for xi in 2:length(X)
		s = α * X[xi] + (1-α) * s
	end
	s
end

# ╔═╡ 27d9ae18-c026-4c10-befe-5439f51d13f0
function double_exp_smoothing(α, β, X)
	if length(X) == 1
		return X[1]
	end
	s = X[1]
	b = X[2] - X[2]
	for xi in 2:length(X)
		stm1 = s
		s = α * X[xi] + (1-α) * s
		b = β * (s - stm1) + (1-β) * b
	end
	b
end

# ╔═╡ b7b31a97-81c5-40d6-a211-6f02489bb845
function triple_exp_smoothing(α, seq)
	s = seq[1]
	for xi in 2:length(seq)
		s = α * seq[xi] + (1-α) * s
	end
	s
end

# ╔═╡ 914c33e6-ff63-4af8-ba2b-4d5999327abb
md"""
## Utility Funcs
- `unit_norm`/`unit_norm!`
- `series_sum`
"""

# ╔═╡ f6b4a7a9-b729-4c3a-9e56-206e66795b77
function unit_norm!(seq)
	seq .= (seq .- minimum(seq))./(maximum(seq) - minimum(seq))
end

# ╔═╡ f5607f9c-033c-48e2-ab64-08d1b4a9821e
function unit_norm(seq)
	(seq .- minimum(seq))./(maximum(seq) - minimum(seq))
end

# ╔═╡ f9c6ee93-b346-46e5-8a0a-a8545dc21306
function series_sum(γ, i)
	s = 0
	for i in 1:i
		s += γ^(i-1)*i
	end
	s
end

# ╔═╡ 64a1c059-6dc7-44b5-8cef-f5d517871aab
md"""
# Cycle World Experiments

- GVF compositional chain. With first GVF as a prediction with c=1 at the head of the cycle.
- Tabular feature representation.
- Each GVF after the first is normalized by (1-\gamma)

"""

# ╔═╡ 1dd3e976-d74b-40bf-ab23-642fc6ccd5ea
md"""
## Experiment Function
"""

# ╔═╡ c06a9fca-e4bc-4748-9970-268786ee1f5a
function cycleworld_experiment!(
		horde,
		env_size, 
		num_steps, 
		lu = TDλ(0.1, 0.9); kwargs...)

	
	env = CycleWorld(env_size, partially_observable=false)
	
	s_t = start!(env)
	
	for step in 1:num_steps
		
		s_tp1, r_tp1, _ = step!(env, 1)
		p_tp1 = predict(horde, s_tp1)
		update!(lu, horde, s_t, s_t, nothing, nothing, s_tp1, s_tp1, r_tp1, p_tp1)

		s_t = copy(s_tp1)
	end
	
	p = [predict(horde, [x]) for x in [1:env_size; 1:env_size]]
	
	p
end

# ╔═╡ 16671204-227f-436c-9e1d-f4f5738df3f9
function cycleworld_experiment(horde_init::Function, args...; kwargs...)
	horde = horde_init()
	p = cycleworld_experiment!(horde, args...; kwargs...)
	horde, p
end

# ╔═╡ 7c77be27-abdf-4625-951d-2601cbac7d84
function cw_run_and_plot(horde_init, args...)
	horde, p = cycleworld_experiment(horde_init, args...)
	plot([plot(getindex.(p, i), legend=nothing) for i in 1:length(horde)]...)
end

# ╔═╡ 18765352-0a28-44b5-a8a3-fa1063e84da3
md"""
## Myopic Discount
"""

# ╔═╡ bb28674b-442e-4c84-bb5e-ba86b8d3c9db
let
	env_size = 10
	num_steps = 100000
	lu = TDλ(0.1, 0.9)
	γ = 0.0
	cw_run_and_plot(env_size, num_steps, TDλ(0.1, 0.9)) do 
		[[GVF(env_size, 
			FeatureCumulant(1), 
			OnPolicy(), 
			ConstantDiscount(γ))];
		[GVF(env_size, 
			PredictionCumulant(i), 
			OnPolicy(), 
			ConstantDiscount(γ)) for i in 1:6]]
	end
end

# ╔═╡ 550e9a33-00d0-4312-849c-6b9c8d49e8c6
md"""
## Constant Discount
"""

# ╔═╡ c4807ce1-0451-4f5c-868f-a8dbc2112919
let	
	env_size = 10
	num_steps = 100000
	lu = TDλ(0.1, 0.9)
	γ = 0.9
	cw_run_and_plot(env_size, num_steps, TDλ(0.1, 0.9)) do 
		[[GVF(env_size, 
			FeatureCumulant(1), 
			OnPolicy(), 
			ConstantDiscount(γ))];
		[GVF(env_size, 
			PredictionCumulant(i), 
			OnPolicy(), 
			ConstantDiscount(γ)) for i in 1:6]]
	end
end

# ╔═╡ 5a116ab2-537e-4eb4-93ff-788ddf741fdf
md"""
## TerminatingDiscount
"""

# ╔═╡ e82378ac-e02a-4425-ab53-a372fa831a40
let		
	env_size = 10
	num_steps = 100000
	lu = TDλ(0.1, 0.9)
	γ = 0.9
	cw_run_and_plot(env_size, num_steps, TDλ(0.1, 0.9)) do 
		[[GVF(env_size, 
			FeatureCumulant(1), 
			OnPolicy(), 
			TerminatingDiscount(γ, 1))];
		[GVF(env_size, 
			PredictionCumulant(i), 
			OnPolicy(), 
			TerminatingDiscount(γ, 1)) for i in 1:6]]
	end
end

# ╔═╡ 24db8a0b-ea04-437d-af63-02709f41d357
md"""
### w/ Threshold Cumulant
"""

# ╔═╡ 04cbe365-e019-4963-a191-68ff02fd13b3
let	
	env_size = 10
	num_steps = 100000
	lu = TDλ(0.1, 0.9)
	γ = 0.9
	cw_run_and_plot(env_size, num_steps, TDλ(0.1, 0.9)) do 
		[[GVF(env_size, 
			FeatureCumulant(1), 
			OnPolicy(), 
			TerminatingDiscount(γ, 1))];
		[GVF(env_size, 
			ThresholdCumulant(PredictionCumulant(i), 0.5), 
			OnPolicy(), 
			TerminatingDiscount(γ, 1)) for i in 1:6]]
	end
end

# ╔═╡ f56773f8-57aa-4157-bc65-dea6bce7f6cc
md"""
# MSO
Visualization

start: $(@bind rng_begin NumberField(1:5:10000)) \
length: $(@bind rng_length NumberField(1:1000, default=200))
"""

# ╔═╡ 77073f41-cde0-42fb-a4b9-9a6ef3285923
data_mso = let
	env = MSO()
	s_0 = start!(env)
	[[s_0[1]]; [step!(env)[1] for i in 1:10000]]
end

# ╔═╡ 9af2f2b9-a095-458b-b602-b2446d9571a5
butterworth_filter(1, unit_norm(data_mso))

# ╔═╡ 04e159db-2740-41ae-b543-8e4f0874fb3b
plot(rng_begin:(rng_length + rng_begin), 
	 data_mso[rng_begin:(rng_length + rng_begin)])

# ╔═╡ d63b2595-c84d-4b14-88b9-b783896655ef
transformed_mso = let
	norm_data_mso = unit_norm(data_mso)
	ret = [norm_data_mso]
	for i in 1:10
		push!(ret, montecarlo_transform_iterate(0.9, ret[end]))
	end
	ret
end

# ╔═╡ cb691a4f-c0cd-490e-9339-65a2c3997071
size(unit_norm(transformed_mso[1][1:100]))

# ╔═╡ f6b19b92-ce33-41a0-a8b1-d239a94604d1
let
	yrng = 1:100
	plts = [plot(tmso[yrng], legend=nothing) for tmso in transformed_mso]
	plot(plts..., size=(1200, 400))
end

# ╔═╡ 162cad05-d12a-482b-b3a0-dad61d494b5b
let
	yrng = 1:1000
	plts = [plot(unit_norm(tmso[yrng]), legend=nothing) 
				for tmso in transformed_mso]
	plot(plts...)
end

# ╔═╡ 186cd369-3f32-45db-886b-7a56c8e876e2
transformed_mso_es = let
	norm_data_mso = unit_norm(data_mso)
	ret = [norm_data_mso]
	for i in 1:15
		push!(ret, 
			exp_smoothing_iterate(ret[end]) do X
				exp_smoothing(0.3, X)
			end
			)
	end
	ret
end

# ╔═╡ 649e5dae-76a1-43f7-a1a7-5776a2cc9792
transformed_mso_es[2]

# ╔═╡ bec89c1d-0818-40f9-be9f-866ec2030db7
let
	yrng = 1:1000
	# plt = plot(data_mso[yrng])
	plts = [plot(tmso[yrng], legend=nothing) 
				for tmso in transformed_mso_es]
	plot(plts...)
end

# ╔═╡ 08406dd9-6299-4b36-aae8-14e5f968a13f
transformed_mso_bw = let
	
	norm_data_mso = unit_norm(data_mso)
	ret = [norm_data_mso]
	for i in 1:10
		push!(ret, 
			  butterworth_filter(10, ret[end], cutoff_freq=0.09))
	end
	ret
end

# ╔═╡ 74b627ee-d38e-4363-88a5-8030b66e846c
let
	yrng = 1:1000
	# plt = plot(data_mso[yrng])
	plts = [plot(tmso[yrng], legend=nothing) 
				for tmso in transformed_mso_bw]
	plot(plts...)
end

# ╔═╡ 2523f136-aab9-4bfc-9c7e-1d5639afc28a
md"""
### All hands plot
"""

# ╔═╡ 175bff6b-e2ab-4ad8-abc1-02ed72d47b08
let
	yrng = 1:1000
	plot(transformed_mso_bw[7][yrng], label="butterworth", lw = 2)
	plot!(unit_norm(data_mso[yrng]), label="unit_norm_mso")
	plot!(unit_norm(transformed_mso[10][yrng]), label="bellmen", lw = 2)
end

# ╔═╡ 0e0c3e72-c469-4ca9-92f9-cd7de610e982
md"""
### Values of γ
"""

# ╔═╡ 175c1427-73e7-4844-9085-3c8d5f087c7e
transformed_mso_gamma = let
	d = Dict{Float64, Any}()
	for γ in 0.0:0.1:0.9
		norm_data_mso = unit_norm(data_mso)
		ret = [norm_data_mso]
		for i in 1:10
			push!(ret, montecarlo_transform_iterate(γ, ret[end]))
		end
		d[γ] = ret
	end
	d
end

# ╔═╡ 528b0f3c-a97b-4e94-94fd-f3c89f7569fb
let
	yrng = 450:550
	plot(unit_norm(transformed_mso_gamma[0.5][10][yrng]), label="bellmen", lw = 2)
	plot!(unit_norm(data_mso[yrng]), label="unit_norm_mso")
end

# ╔═╡ 7accaef0-534b-41d8-a879-1727f96823f2
md"""
# Sine Wave Analysis
"""

# ╔═╡ 403fc0c1-74d9-4ea6-87a8-270a91ae73d7
sw_gammas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.995]

# ╔═╡ d313e99b-61fe-45c8-8449-ba56fae1521c
function create_sine_dataset(N, A₁, f, ϕ₁)
	s = (n) -> A₁ * sin(2*π*f*n + ϕ₁)
	[s(i-1) for i in 1:N]
end

# ╔═╡ 60092c35-e38b-4534-8662-7d2b7a013dfe
plot(create_sine_dataset(1000, 1, 0.01, 0))

# ╔═╡ bd3bd8ae-14af-4bce-984c-08e0860cf35e
sine_analysis_mc = let
	d = Dict{Float64, Vector{Vector{Float64}}}()
	for γ ∈ sw_gammas
		norm_data_mso = create_sine_dataset(10000, 1, 0.01, 0)
		ret = [norm_data_mso]
		for i in 1:10
			push!(ret, montecarlo_transform_iterate(γ, ret[end]))
		end
		d[γ] = ret
	end
	d
end

# ╔═╡ 612b1e9d-80b6-4583-b742-c2de8c3d56f8
let
	yrng = 1:1000
	plt = plot(sine_analysis_mc[0.0][1][yrng])
	for γ in sw_gammas
		plot!(plt, sine_analysis_mc[γ][2][yrng], label=γ)
	end
	plt
end

# ╔═╡ f9327c7e-44fc-4938-a867-55e295b5bb26
let
	plt = plot()
	for i in 1:length(sine_analysis_mc[0.0])
		A = [maximum(sine_analysis_mc[γ][i]) for γ in sw_gammas]
		plot!(plt, sw_gammas, A, yaxis=:log)
	end
	plt
end

# ╔═╡ 7f242537-65e9-455c-ae50-38cedd0fe65e
md"""
# Other DataSets
"""

# ╔═╡ 152658e4-12ea-4bed-a442-8af4f8c445fd
DataFrames.subset(RDatasets.datasets(), :Dataset => a -> a .== "macro")

# ╔═╡ 689533b8-10de-4cf0-a5a8-21cdc34102cc
RDatasets.dataset("Zelig", "macro")[1:25, 3]

# ╔═╡ 65b28a24-3e42-4e15-83f7-d0559fe68ab6
begin
	sm = PyCall.pyimport("statsmodels.api")
	dta = collect(sm.datasets.macrodata.load_pandas()["data"]["infl"])
end

# ╔═╡ e2b3b3ef-64ca-4116-9a0c-6c89fded671d
let
	plot(butterworth_filter(10, unit_norm(dta); cutoff_freq=0.1))
	plot!(unit_norm(dta))
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
FourierAnalysis = "e7e9c730-dc46-11e9-3633-f1ab55cc17e1"
LoopVectorization = "bdcacae8-1622-11e9-2a5c-532679323890"
MinimalRLCore = "4557a151-568a-41c4-844f-9d8069264cea"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
RDatasets = "ce6b1742-4840-55fa-b093-852dadbb1d8b"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Tullio = "bc48ee85-29a4-5162-ae0b-a64e1601d4bc"

[compat]
DSP = "~0.7.4"
DataFrames = "~1.3.2"
FourierAnalysis = "~1.2.0"
LoopVectorization = "~0.12.101"
MinimalRLCore = "~0.2.1"
Plots = "~1.23.6"
PlutoUI = "~0.7.34"
PyCall = "~1.93.0"
RDatasets = "~0.7.7"
StatsPlots = "~0.14.30"
Tullio = "~0.3.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.1"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "051c95d6836228d120f5f4b984dd5aba1624f716"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "0.5.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra"]
git-tree-sha1 = "2ff92b71ba1747c5fdd541f8fc87736d82f40ec9"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.4.0"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "1ee88c4c76caa995a885dc2f22a5d548dfbbc0ba"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.2.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "5e98d6a6aa92e5758c4d58501b7bf23732699fa3"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.2"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CPUSummary]]
deps = ["Hwloc", "IfElse", "Static"]
git-tree-sha1 = "ba19d1c8ff6b9c680015033c66802dd817a9cf39"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.1.7"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "9519274b50500b8029973d241d32cfbf0b127d97"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.2"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "d0b3f8b4ad16cb0a2988c6788646a5e6a17b6b1b"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.0.5"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "c308f209870fdbd84cb20332b6dfaf14bf3387f8"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f9982ef575e19b0e5c7a98c6e75ee496c0f73a93"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.12.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.CloseOpenIntervals]]
deps = ["ArrayInterface", "Static"]
git-tree-sha1 = "7b8f09d58294dc8aa13d91a8544b37c8a1dcbc06"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.4"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Luxor", "Random"]
git-tree-sha1 = "5b7d2a8b53c68dfdbce545e957a3b5cc4da80b01"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.17.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.CommonRLInterface]]
deps = ["MacroTools"]
git-tree-sha1 = "21de56ebf28c262651e682f7fe614d44623dc087"
uuid = "d842c3ba-07a1-494f-bbec-f5741b0a3e98"
version = "0.3.1"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "6cdc8832ba11c7695f494c9d9a1c31e90959ce0f"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.6.0"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DSP]]
deps = ["Compat", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "fe2287966e085df821c0694df32d32b6311c6f4c"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.7.4"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "ae02104e835f219b8930c7664b8012c93475c340"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.2"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "84083a5136b6abf426174a58325ffd159dd6d94f"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.9.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "38012bf3553d01255e83928eec9c998e19adfddf"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.48"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ae13fcbc7ab8f16b0856729b050ef0c446aa3492"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.4+0"

[[deps.ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "IntelOpenMP_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Reexport"]
git-tree-sha1 = "8fda0934cb99db617171f7296dc361f4d6fa5424"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.3.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "80ced645013a5dbdc52cf70329399c35ce007fae"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.13.0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "04d13bfa8ef11720c24e4d840c0033d145537df7"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.17"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "deed294cde3de20ae0b2e0355a6c4e1c6a5ceffc"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.8"

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

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "1bd6fc0c344fc0cbee1f42f8d2e7ec8253dda2d2"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.25"

[[deps.FourierAnalysis]]
deps = ["AbstractFFTs", "DSP", "FFTW", "LinearAlgebra", "PosDefManifold", "RecipesBase", "Statistics"]
git-tree-sha1 = "e6f04b14dd9ee49a8245d5c3060ffad3539246f8"
uuid = "e7e9c730-dc46-11e9-3633-f1ab55cc17e1"
version = "1.2.0"

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

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "30f2b340c2fff8410d89bfcdc9c0a6dd661ac5f7"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.62.1"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "aa22e1ee9e722f1da183eb33370df4c1aeb6c2cd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.63.1+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "1c5a84319923bea76fa145d49e93aa4394c73fc2"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.1"

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
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "3965a3216446a6b020f0d48f1ba94ef9ec01720d"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.6"

[[deps.Hwloc]]
deps = ["Hwloc_jll"]
git-tree-sha1 = "92d99146066c5c6888d5a3abc871e6a214388b91"
uuid = "0e44f5e4-bd66-52a0-8798-143a42290a1d"
version = "2.0.0"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d8bccde6fc8300703673ef9e1383b11403ac1313"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.7.0+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "61feba885fac3a407465726d0c330b3055df897f"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.2"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "b15fc0a95c564ca2e0a7ae12c1f095ca848ceb31"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.5"

[[deps.Intervals]]
deps = ["Dates", "Printf", "RecipesBase", "Serialization", "TimeZones"]
git-tree-sha1 = "323a38ed1952d30586d0fe03412cde9399d3618b"
uuid = "d8418881-c3e1-53bb-8760-2df7ec849ed5"
version = "1.5.0"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.Juno]]
deps = ["Base64", "Logging", "Media", "Profile"]
git-tree-sha1 = "07cb43290a840908a771552911a6274bc6c072c7"
uuid = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
version = "0.8.4"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

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
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static"]
git-tree-sha1 = "6dd77ee76188b0365f7d882d674b95796076fa2c"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.5"

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

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
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Librsvg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pango_jll", "Pkg", "gdk_pixbuf_jll"]
git-tree-sha1 = "25d5e6b4eb3558613ace1c67d6a871420bfca527"
uuid = "925c91fb-5dd6-59dd-8e8c-345e74382d89"
version = "2.52.4+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

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
git-tree-sha1 = "e5718a00af0ab9756305a0392832c8952c7426c1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.6"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "ChainRulesCore", "CloseOpenIntervals", "DocStringExtensions", "ForwardDiff", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "SIMDDualNumbers", "SLEEFPirates", "SpecialFunctions", "Static", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "67c0dfeae307972b50009ce220aae5684ea852d1"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.101"

[[deps.Luxor]]
deps = ["Base64", "Cairo", "Colors", "Dates", "FFMPEG", "FileIO", "Juno", "LaTeXStrings", "Random", "Requires", "Rsvg"]
git-tree-sha1 = "81a4fd2c618ba952feec85e4236f36c7a5660393"
uuid = "ae8d54c2-7ccd-5906-9d76-62fc9837b5bc"
version = "3.0.0"

[[deps.MKL_jll]]
deps = ["IntelOpenMP_jll", "Libdl", "Pkg"]
git-tree-sha1 = "eb540ede3aabb8284cb482aa41d00d6ca850b1f8"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2020.2.254+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Media]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "75a54abd10709c01f1b86b84ec225d26e840ed58"
uuid = "e89f7d12-3494-54d1-8411-f7d8b9ae1f27"
version = "0.5.0"

[[deps.MinimalRLCore]]
deps = ["CommonRLInterface", "Lazy", "LinearAlgebra", "Random", "StatsBase"]
git-tree-sha1 = "0e13b3968a1247f5578d104f44c74545678c108a"
uuid = "4557a151-568a-41c4-844f-9d8069264cea"
version = "0.2.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.Mocking]]
deps = ["Compat", "ExprTools"]
git-tree-sha1 = "29714d0a7a8083bba8427a4fbfb00a540c681ce7"
uuid = "78c3b35d-d492-501b-9361-3d52fe80e533"
version = "0.7.3"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "8d958ff1854b166003238fe191ec34b9d592860a"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.8.0"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "842b5ccd156e432f369b204bb704fd4020e383ac"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "0.3.3"

[[deps.NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "16baacfdc8758bc374882566c9187e785e85c2f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.9"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "648107615c15d4e09f7eca16307bc821c1f718d8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.13+0"

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

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ee26b350276c51697c9c2d88a072b339f9f03d73"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.5"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a121dfbba67c94a5bec9dde613c3d0cbcf3a12b"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.50.3+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0b5cfbb704034b5b4c1869e36634438a047df065"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "6f1b25e8ea06279b5689263cc538f51331d7ca17"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.1.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun"]
git-tree-sha1 = "0d185e8c33401084cab546a756b387b15f76720c"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.23.6"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8979e9802b4ac3d58c503a20f2824ad67f9074dd"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.34"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "0bc9e1a21ba066335a5207ac031ee41f72615181"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.1.3"

[[deps.Polynomials]]
deps = ["Intervals", "LinearAlgebra", "MutableArithmetics", "RecipesBase"]
git-tree-sha1 = "f184bc53e9add8c737e50fa82885bc3f7d70f628"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "2.0.24"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "db3a23166af8aebf4db5ef87ac5b00d36eb771e2"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.0"

[[deps.PosDefManifold]]
deps = ["LinearAlgebra", "Statistics"]
git-tree-sha1 = "2010295335c00d7465571c6ee5e1eac5dc4a25e3"
uuid = "f45a3650-5c51-11e9-1e9a-133aa5e309cf"
version = "0.4.9"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "2cf929d64681236a2e074ffafb8d568733d2e6af"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.3"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "71fd4022ecd0c6d20180e23ff1b3e05a143959c2"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.93.0"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.RData]]
deps = ["CategoricalArrays", "CodecZlib", "DataFrames", "Dates", "FileIO", "Requires", "TimeZones", "Unicode"]
git-tree-sha1 = "19e47a495dfb7240eb44dc6971d660f7e4244a72"
uuid = "df47a6cb-8c03-5eed-afd8-b6050d6c41da"
version = "0.8.3"

[[deps.RDatasets]]
deps = ["CSV", "CodecZlib", "DataFrames", "FileIO", "Printf", "RData", "Reexport"]
git-tree-sha1 = "2720e6f6afb3e562ccb70a6b62f8f308ff810333"
uuid = "ce6b1742-4840-55fa-b093-852dadbb1d8b"
version = "0.7.7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

[[deps.RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.Rsvg]]
deps = ["Cairo", "Glib_jll", "Librsvg_jll"]
git-tree-sha1 = "3d3dc66eb46568fb3a5259034bfc752a0eb0c686"
uuid = "c4c386cf-5103-5370-be45-f3a111cca3b8"
version = "1.0.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.SIMDDualNumbers]]
deps = ["ForwardDiff", "IfElse", "SLEEFPirates", "VectorizationBase"]
git-tree-sha1 = "62c2da6eb66de8bb88081d20528647140d4daa0e"
uuid = "3cdde19b-5bb0-4aaf-8931-af3e248e098b"
version = "0.1.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "1410aad1c6b35862573c01b96cd1f6dbe3979994"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.28"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "15dfe6b103c2a993be24404124b8791a09460983"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.11"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "8d0c8e3d0ff211d9ff4a0c2307d876c99d10bdf1"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.2"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "7f5a513baec6f122401abfc8e9c074fdac54f6c1"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.4.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "a635a9333989a094bddc9f940c04c549cd66afcf"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.3.4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
git-tree-sha1 = "d88665adc9bcf45903013af0982e2fd05ae3d0a6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "51383f2d367eb3b444c961d485c565e4c0cf4ba0"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.14"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f35e1879a71cca95f4826a14cdbf0b9e253ed918"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.15"

[[deps.StatsPlots]]
deps = ["Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "e1e5ed9669d5521d4bbdd4fab9f0945a0ffceba2"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.30"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "d21f2c564b21a202f4677c0fba5b5ee431058544"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.4"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "bb1064c9a84c52e277f1096cf41434b675cd368b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "884539ba8c4584a3a8173cb4ee7b61049955b79c"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.4.7"

[[deps.TimeZones]]
deps = ["Dates", "Downloads", "InlineStrings", "LazyArtifacts", "Mocking", "Printf", "RecipesBase", "Serialization", "Unicode"]
git-tree-sha1 = "0f1017f68dc25f1a0cb99f4988f78fe4f2e7955f"
uuid = "f269a46b-ccf7-5d73-abea-4c690281aa53"
version = "1.7.1"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.Tullio]]
deps = ["ChainRulesCore", "DiffRules", "LinearAlgebra", "Requires"]
git-tree-sha1 = "7830c974acc69437a3fee35dd7b510a74cbc862d"
uuid = "bc48ee85-29a4-5162-ae0b-a64e1601d4bc"
version = "0.3.3"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "Hwloc", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static"]
git-tree-sha1 = "e9a35d501b24c127af57ca5228bcfb806eda7507"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.24"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "66d72dc6fcc86352f01676e8f0f698562e60510f"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.23.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "c69f9da3ff2f4f02e811c3323c22e5dfcb584cfa"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.1"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "505c31f585405fc375d99d02588f6ceaba791241"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.5"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

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

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.gdk_pixbuf_jll]]
deps = ["Artifacts", "Glib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Xorg_libX11_jll", "libpng_jll"]
git-tree-sha1 = "c23323cd30d60941f8c68419a70905d9bdd92808"
uuid = "da03df04-f53b-5353-a52f-6a8b0620ced0"
version = "2.42.6+1"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

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

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

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
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─729faad8-176f-4b6b-af97-3e321c7be3e0
# ╠═ebe424a8-f385-46ee-80e9-12b99bb2e691
# ╠═098f4309-d067-4d4e-be6c-8dacdc059b89
# ╠═b4d08bb8-ea59-11eb-1df0-93d1149fcf0a
# ╠═5590501e-eab5-4d08-a81f-b1f5a2b91a08
# ╠═1df5d910-1a95-4049-906f-770eb3a7990a
# ╠═84c1f22f-5a42-45de-86f0-3729922336e1
# ╠═64848d48-09a0-4e13-a318-877b66a4f5e6
# ╟─4f5da656-f619-416f-9421-0a21fca291af
# ╟─cefce664-1c9e-435e-b6bc-1be062d95e1b
# ╟─76453841-7bec-4fd4-bdd1-1ea185f26e13
# ╠═ff6da4df-5c4f-4167-9fe9-e965d63165bf
# ╟─118bac06-bd77-4ab3-a5fe-f9a7590cb781
# ╟─289d39df-98c3-4572-8347-7ecf704f4be5
# ╠═285488f4-1e74-4e15-816b-077722c4677b
# ╟─0dbae625-1a58-4fc9-a115-84411becdcc0
# ╠═9f03c3bc-6fac-4e5d-8b39-7e0b5a891e71
# ╟─9ce73bf3-1cdb-4a0f-bbab-b7c331a7b7fe
# ╠═3dbe0930-8fb7-4dea-b2e3-0110a28a7249
# ╟─ab28215c-74c0-4722-b96d-5c3862dab13d
# ╠═6b865983-65d2-478f-b891-bdaf0092e2ce
# ╟─f2ab0915-4752-4248-bb65-d90b3f929539
# ╟─2b14f50c-3b46-42ad-a01c-5e47d31914e4
# ╠═131b2d9b-711b-4cea-bab1-03f0ef68f5a9
# ╟─91fdbc6f-479c-4a79-848b-b0a83268348b
# ╟─56f4136d-3e82-47ac-91a3-48f7331ef7c7
# ╠═725c9586-615d-4d27-8a2f-fe2760aeaedc
# ╟─7cbb8f85-1fd3-4c0d-a081-0d9c487227e6
# ╠═4d7bcae2-e7dd-4aa4-84e5-6c529be7c2b4
# ╟─85392583-6481-4a77-96c0-30f136e08299
# ╟─2042308c-a555-4ab2-8eec-75925659b504
# ╠═2410a3a2-f1d6-4edf-ae52-66d43816093b
# ╠═3d3dd633-4f9f-4cad-8018-dfe2bebffa5b
# ╠═b40e514c-04d0-4690-ad89-b81761634bf4
# ╠═2b3f8086-9f7d-4fa7-bfb9-a9a5746fcd5d
# ╟─010f3a6c-e331-4609-b180-638914171c18
# ╟─0a9dbac5-195b-4c78-a0e4-62f6727e4fce
# ╟─0ce6eaec-1c2f-478b-96fd-fe4965517e46
# ╠═ae9413e4-918e-4405-928d-85d64bcebd17
# ╟─adf1838e-bc3a-44de-baaa-de33e77c3296
# ╟─66d0475a-cfd5-4805-b79f-298645fe9592
# ╟─83128a03-0274-432d-b307-960d65502dae
# ╟─e951e11c-fef2-4bd6-85dd-0c5186bf1771
# ╟─8126a898-1d98-4013-91a6-8acba476556a
# ╠═81f79509-8f08-4df3-8074-2711e6fbdca7
# ╟─1568abbd-856f-46bb-bf30-86ee3e6553c6
# ╟─91957826-379f-4db4-9a61-fc7c5fa667b1
# ╠═931db888-1f65-4ac8-965a-e3fa12672ea4
# ╠═bc620363-6bf2-4c06-b066-f68cbea8a290
# ╟─611e00d5-cf92-4e0b-9504-8b611b88a897
# ╠═917e3c4f-75c7-4759-81d5-a0c6823d784a
# ╠═8850c353-20ba-4181-ac5a-70ca86e15066
# ╠═9af2f2b9-a095-458b-b602-b2446d9571a5
# ╟─4a1fe221-fb84-4aa4-9691-3dc2fa3893c3
# ╠═8f094be4-9228-49dd-9b77-89917e4825cf
# ╠═c0da6083-792f-4d1f-99c4-6a69c09d01d2
# ╠═27d9ae18-c026-4c10-befe-5439f51d13f0
# ╠═b7b31a97-81c5-40d6-a211-6f02489bb845
# ╟─914c33e6-ff63-4af8-ba2b-4d5999327abb
# ╠═f6b4a7a9-b729-4c3a-9e56-206e66795b77
# ╠═f5607f9c-033c-48e2-ab64-08d1b4a9821e
# ╠═f9c6ee93-b346-46e5-8a0a-a8545dc21306
# ╟─64a1c059-6dc7-44b5-8cef-f5d517871aab
# ╟─1dd3e976-d74b-40bf-ab23-642fc6ccd5ea
# ╠═16671204-227f-436c-9e1d-f4f5738df3f9
# ╠═c06a9fca-e4bc-4748-9970-268786ee1f5a
# ╠═7c77be27-abdf-4625-951d-2601cbac7d84
# ╟─18765352-0a28-44b5-a8a3-fa1063e84da3
# ╠═bb28674b-442e-4c84-bb5e-ba86b8d3c9db
# ╟─550e9a33-00d0-4312-849c-6b9c8d49e8c6
# ╠═c4807ce1-0451-4f5c-868f-a8dbc2112919
# ╟─5a116ab2-537e-4eb4-93ff-788ddf741fdf
# ╠═e82378ac-e02a-4425-ab53-a372fa831a40
# ╟─24db8a0b-ea04-437d-af63-02709f41d357
# ╠═04cbe365-e019-4963-a191-68ff02fd13b3
# ╟─f56773f8-57aa-4157-bc65-dea6bce7f6cc
# ╟─77073f41-cde0-42fb-a4b9-9a6ef3285923
# ╠═04e159db-2740-41ae-b543-8e4f0874fb3b
# ╠═d63b2595-c84d-4b14-88b9-b783896655ef
# ╠═cb691a4f-c0cd-490e-9339-65a2c3997071
# ╠═f6b19b92-ce33-41a0-a8b1-d239a94604d1
# ╠═162cad05-d12a-482b-b3a0-dad61d494b5b
# ╠═186cd369-3f32-45db-886b-7a56c8e876e2
# ╠═649e5dae-76a1-43f7-a1a7-5776a2cc9792
# ╠═bec89c1d-0818-40f9-be9f-866ec2030db7
# ╠═08406dd9-6299-4b36-aae8-14e5f968a13f
# ╠═74b627ee-d38e-4363-88a5-8030b66e846c
# ╟─2523f136-aab9-4bfc-9c7e-1d5639afc28a
# ╠═175bff6b-e2ab-4ad8-abc1-02ed72d47b08
# ╟─0e0c3e72-c469-4ca9-92f9-cd7de610e982
# ╠═175c1427-73e7-4844-9085-3c8d5f087c7e
# ╠═528b0f3c-a97b-4e94-94fd-f3c89f7569fb
# ╟─7accaef0-534b-41d8-a879-1727f96823f2
# ╠═403fc0c1-74d9-4ea6-87a8-270a91ae73d7
# ╠═d313e99b-61fe-45c8-8449-ba56fae1521c
# ╠═60092c35-e38b-4534-8662-7d2b7a013dfe
# ╠═bd3bd8ae-14af-4bce-984c-08e0860cf35e
# ╠═612b1e9d-80b6-4583-b742-c2de8c3d56f8
# ╠═f9327c7e-44fc-4938-a867-55e295b5bb26
# ╟─7f242537-65e9-455c-ae50-38cedd0fe65e
# ╠═152658e4-12ea-4bed-a442-8af4f8c445fd
# ╠═689533b8-10de-4cf0-a5a8-21cdc34102cc
# ╠═65b28a24-3e42-4e15-83f7-d0559fe68ab6
# ╠═e2b3b3ef-64ca-4116-9a0c-6c89fded671d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
