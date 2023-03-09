### A Pluto.jl notebook ###
# v0.19.22

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
using Random, Revise

# ╔═╡ 098f4309-d067-4d4e-be6c-8dacdc059b89
using PlutoUI, StatsPlots, Plots

# ╔═╡ 44ccdaa4-d952-4367-8bf0-15c8841a253d
using ProgressLogging

# ╔═╡ b4d08bb8-ea59-11eb-1df0-93d1149fcf0a
using MinimalRLCore

# ╔═╡ 917e3c4f-75c7-4759-81d5-a0c6823d784a
using DSP

# ╔═╡ a0c15db4-d43f-40ce-ab35-e7583b7c0dd4


# ╔═╡ 729faad8-176f-4b6b-af97-3e321c7be3e0
md"""
- [Time As a Variable: Time-Series Analysis](https://www.oreilly.com/library/view/data-analysis-with/9781449389802/ch04.html)
- [A Very Short Course on Time Series Analysis](https://bookdown.org/rdpeng/timeseriesbook/filtering-time-series.html)
"""

# ╔═╡ 5590501e-eab5-4d08-a81f-b1f5a2b91a08
import DataFrames

# ╔═╡ 1df5d910-1a95-4049-906f-770eb3a7990a
import FourierAnalysis

# ╔═╡ 223375db-94c7-4ac1-bdae-0d6856e2f492
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

# ╔═╡ 24457b9c-64a9-43a4-b3c1-b1f5af7ed823
function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end

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
	ret = sum(view(w, x))
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
        view(z, x_t) .+= 1
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
# Data
"""

# ╔═╡ 3cbe38c5-11a8-4d99-b5b7-613eced6139b
md"""
## Cycle World
"""

# ╔═╡ 512344a1-e3ba-4a6f-ab87-b5ef344c413d
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

# ╔═╡ 2c639be3-3867-45e2-80f3-3c0890529f1c
function MinimalRLCore.reset!(env::CycleWorld, rng=nothing; kwargs...)
		env.agent_state = 0
	end

# ╔═╡ ba06366c-13c7-4112-a8f9-998346c49910
MinimalRLCore.get_actions(env::CycleWorld) = Set(1:1)

# ╔═╡ 5ba27c96-e3a1-4ebd-b210-72bf4262cdfb
function MinimalRLCore.environment_step!(
			env::CycleWorld,
			action, 
			rng=nothing; 
			kwargs...)
		env.agent_state = (env.agent_state + 1) % env.chain_length
	end

# ╔═╡ 4bf138a4-d5cd-401c-a627-4633b224f4d7
MinimalRLCore.get_reward(env::CycleWorld) = 0 # -> get the reward of the environment

# ╔═╡ 37cf17e2-7796-4342-8ca6-cd8bf9a4ed9b
fully_observable_state(env::CycleWorld) = [env.agent_state+1]

# ╔═╡ 9192754f-a710-4f8e-8c0b-a11de74bc240
function partially_observable_state(env::CycleWorld)
	state = zeros(1)
	if env.agent_state == 0
		state[1] = 1
	end
	return state
end

# ╔═╡ e2eeb4ec-06d4-4731-ad25-b9c956457d11
function partially_observable_state(state::Int)
	state = zeros(1)	
	if state == 0
		state[1] = 1
	end
	return state
end

# ╔═╡ fc965f74-158c-445f-b822-b00840e5de1b
function MinimalRLCore.get_state(env::CycleWorld) # -> get state of agent
		if env.partially_observable
			return partially_observable_state(env)
		else
			return fully_observable_state(env)
		end
	end

# ╔═╡ c8b3fe86-4433-4d08-b303-8cd9d91f9953
function MinimalRLCore.is_terminal(env::CycleWorld) # -> determines if the agent_state is terminal
		return false
	end

# ╔═╡ 52bb70f3-6309-41c6-a65c-6720db2b4a05
function Base.show(io::IO, env::CycleWorld)
		model = fill("0", env.chain_length)
		model[1] = "1"
		println(io, join(model, ' '))
		model = fill("-", env.chain_length)
		model[env.agent_state + 1] = "^"
		println(io, join(model, ' '))
		# println(env.agent_state)
	end

# ╔═╡ 35ed964f-819e-4f46-a531-3da2689356f4
md"""
## MSO
"""

# ╔═╡ 792c90c4-f66c-4695-8897-f7bb990ca31e
begin
	mutable struct MSO <: MinimalRLCore.AbstractEnvironment
		θ::Int
		Ω::Vector{Float64}
		state::Vector{Float64}
	end
	MSO() = MSO(1, [0.2, 0.311, 0.42, 0.51], [0.0])

end

# ╔═╡ fbe6c146-87d2-4ee6-b9b9-1818e80e6959
function MinimalRLCore.step!(self::MSO)
		self.state[1] = sum([sin(self.θ*ω) for ω in self.Ω])
		self.θ += 1
		return self.state
	end

# ╔═╡ e7c38a2a-343a-48dc-bce4-0fe5df04c65d
function MinimalRLCore.start!(self::MSO)
		self.θ = 1
		return step!(self)
	end

# ╔═╡ e4deba4c-f5a9-4736-b95c-2ef51e8e8f3e
MinimalRLCore.get_state(self::MSO) = self.state

# ╔═╡ 69b43c3e-e180-449d-a1fe-91d686c183e8
get_num_features(self::MSO) = 1

# ╔═╡ 79ebfb96-5bb2-4337-839e-bc1ac81cb0a7
get_num_targets(self::MSO) = 1

# ╔═╡ 74d3635c-304d-40f6-a1b4-f8911411d933
md"""
## Critterbot
"""

# ╔═╡ 650de732-e312-41d5-938c-beb5e330ee0f
import HDF5

# ╔═╡ c88e09f3-88c4-4e27-837c-bce455447e99
CritterbotUtils = ingredients("utils/CritterbotUtils.jl").CritterbotUtils

# ╔═╡ 94eaa0f6-cbd6-4d10-a24c-f6296048633b
critterbot_data = let
	col_names = CritterbotUtils.sensor_names()[CritterbotUtils.relevant_sensors_idx()]
	cb_data = CritterbotUtils.relevant_sensors()
	DataFrames.DataFrame(;(Symbol(n)=>d for (n, d) in zip(col_names, eachslice(cb_data, dims=2)))...)
end

# ╔═╡ dd1c2f72-379a-4013-a5fc-adf717905b74
md"## Random Intervals "

# ╔═╡ 5a79876d-d7bf-4829-9658-4ea4477b0349
let
	data = zeros(Int, 10000)
	p1 = 1.0
	p2 = 0.05
	rng = Random.Xoshiro(4)
	cur_p = 0.05
	for i in eachindex(data)
		p = rand(rng)
		if p < cur_p
			if cur_p == p2
				cur_p = p1
			end
			data[i] = 1
		end
		cur_p = max(p2, cur_p - 0.2)

		p3 = rand(rng)
		if p3 < 0.05
			data[i] = 1
		end
	end
	plot(data[1:1000])
end

# ╔═╡ d3ed34bb-3f5d-4565-8465-dc1fa0cc1f9b
random_data = let
	data = zeros(Float64, 10000)
	rng = Random.Xoshiro(4)
	cur_p = 0.05
	for i in eachindex(data)
		p = rand(rng)
		if p < cur_p
			data[i] = 1
		end
	end
	plot(data[1:1000])
	data
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

# ╔═╡ 62fde1ef-c6db-4ba7-b484-e8c9ac530f69
function montecarlo_transform_iterate(γ, seq)
	rets = zeros(eltype(seq), length(seq)+1)
	# for (r_idx, r) in Iterators.reverse(enumerate(seq))
	for t in length(seq)-1:-1:1
		rets[t] = seq[t+1] + γ*rets[t+1]
	end
	rets[1:end-1]
end

# ╔═╡ 3a4d265e-d226-480f-ba8d-6064f601c6f1
function montecarlo_transform_iterate(γ::Function, seq)
	rets = zeros(eltype(seq), length(seq)+1)
	for t in length(seq)-1:-1:1
		rets[t] = seq[t+1] + γ(seq[t+1], t+1)*rets[t+1]
		# end
	end
	rets[1:end-1]
end

# ╔═╡ e7643fc3-f032-47cb-b717-97516b7a696b
let
	x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.6, 0.8, 0.9, 0.91, 0.91, 0.8, 0.5, 0.1, 0.0, 0.0]
	montecarlo_transform_iterate(x) do x, t
		x > 0.9 ? 0.0 : 0.9
	end
end

# ╔═╡ 611e00d5-cf92-4e0b-9504-8b611b88a897
md"""
## Butterworth Filters
"""

# ╔═╡ 46c3ba82-c56e-4ee1-9fad-2523137ba84d
md"""
Some resources on time-series filters.

- [Time As a Variable: Time-Series Analysis](https://www.oreilly.com/library/view/data-analysis-with/9781449389802/ch04.html)
- [A Very Short Course on Time Series Analysis](https://bookdown.org/rdpeng/timeseriesbook/filtering-time-series.html)
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

# ╔═╡ d5cc227c-f5f7-47aa-9027-597879fc02df
md"""
# Random Cumulant Experiments
"""

# ╔═╡ eaeb04c5-5371-4811-849e-e3805768d1c2
function plot_gamma_arb_mult(f_γ, t; T=30)
	rng = Random.Xoshiro(3)
	x_top=T

	# t = 15
	
	cw_obs = fill(0.0, x_top + 1)
	for i in 1:(x_top+1)
		cw_obs[i] = rand(rng)
	end
	plt = bar(
		1:x_top+1, 
		cw_obs, 
		bar_width=0.04, 
		xrange=(0, x_top+2.1),
		yrange=(0.0, 1.1), 
		legend=false, 
		grid=false, 
		tick_dir=:in,
		yformatter=(a)->"", xformatter=(a)->"", 
		yticks=false)
	
	scatter!(plt, 1:x_top+1, cw_obs, color=color_scheme[2])

	plt2 = bar(
		1:x_top+1,
		(x)->f_γ(t, x),
		bar_width=0.04,
		xrange=(0, x_top+2.1),
		yrange=(0.0, 1.1), 
		# lw=3,
		legend=false, 
		grid=false, 
		axis=false,
		yformatter=(a)->"", xformatter=(a)->"", 
		yticks=false, 
		color=:black)

	scatter!(plt2, 1:x_top+1, (x)->f_γ(t, x), color=color_scheme[1])

	plt_both = plot(plt2, plt, layout=(2,1))


	plt = bar(
		1:x_top+1,
		(x)->f_γ(t, x) * cw_obs[x], 
		bar_width=0.04, 
		xrange=(0, x_top+2.1), 
		yrange=(0.0, 1.1), 
		legend=false, 
		grid=false, 
		tick_dir=:in,
		yformatter=(a)->"", xformatter=(a)->"", 
		yticks=false)
	
	scatter!(plt, 
			 1:x_top+1, 
			 (x)->f_γ(t, x) * cw_obs[x], 
			 color=color_scheme[5])

	plt_both, plt

end

# ╔═╡ 99e4a6e9-caa4-436a-b05d-8059889ca95e
plot_gamma_mult(t) = plot_gamma_arb_mult(t) do t, x
	x < t+1 ? 0.0 : 0.9 ^ (x - (t+1))
end


# ╔═╡ ad61416a-3c81-44a9-a2b1-26879b9499c7
function γⁿ_closed_form(γ, n, k)
	if k < n
		zero(γ)
	elseif n == 1 #NOTE: Deal with prod, closed form works with n=1 as 0!=1
		γ^(k-1)
	else
		prod((k-i) for i in 1:(n-1))/factorial(n-1) * γ^(k-n)
	end
end

# ╔═╡ 6a63a5b5-f42f-4e97-9d43-eced19c01cf9
@bind _rand_t PlutoUI.Slider(0:30)

# ╔═╡ 51255b81-9bab-4614-a6fe-22c9db4d40d8
let
	plts = plot_gamma_mult(_rand_t)
	plot(plts[1], plts[2], size=(700, 300))
end

# ╔═╡ ae6c39ee-2366-41d8-a576-b43e1c441d6b
@bind _rand_n_t PlutoUI.Slider(1:30)

# ╔═╡ f20bd192-bc92-4a74-8c24-50b32f83b604
@bind _rand_n PlutoUI.NumberField(1:20)

# ╔═╡ 8111936a-2c3d-4f2c-8f6c-e25feb60ec72
let
	n = _rand_n
	γ = 0.9
	h = Int(round(1/(1-γ)))
	plts = plot_gamma_arb_mult(_rand_n_t; T=50) do t, x
		# x < t+1 ? 0.0 : 0.9 ^ (x - (t+1))
		γⁿ_closed_form(γ, n, x-t) / (n == 1 ? 1.0 : max(γⁿ_closed_form(γ, n, (n-1)*h), γⁿ_closed_form(γ, n, (n-1)*h+1)))
	end
	plot(plts[1], plts[2], size=(700, 300))
end

# ╔═╡ 04be37d5-9934-481b-98e1-da94dba09e87
let
	local_plot(n, t) = begin
		# n = _rand_n
		γ = 0.9
		h = Int(round(1/(1-γ)))
		plts = plot_gamma_arb_mult(t; T=50) do t, x
			# x < t+1 ? 0.0 : 0.9 ^ (x - (t+1))
			γⁿ_closed_form(γ, n, x-t) / (n == 1 ? 1.0 : max(γⁿ_closed_form(γ, n, (n-1)*h), γⁿ_closed_form(γ, n, (n-1)*h+1)))
		end
		plot(plts[1], plts[2], size=(700, 300))
	end

	if !isdir("nplot_rand_data")
		mkdir("nplot_rand_data")
	end
	
	for n in 1:5
		for t in 1:30
			plt = local_plot(n, t)
			savefig(plt, "nplot_rand_data/n=$(n),t=$(t).pdf")
		end
	end

end

# ╔═╡ a88b0b03-f496-4f70-8271-5a371d0ce769
md"""
# Single Cumulant Experiment
"""

# ╔═╡ 8049b4cd-459c-4ae1-8d28-b357d073a527
function plot_cycleworld_gamma_arb_mult(f_γ, t; T=30)
	rng = Random.Xoshiro(3)
	x_top=T

	# t = 15
	
	cw_obs = fill(0.0, x_top + 1)
	for i in 1:(x_top+1)
		if i % 10 == 1
			cw_obs[i] = 1
		end
	end
	plt = bar(
		1:x_top+1, 
		cw_obs, 
		bar_width=0.04, 
		xrange=(0, x_top+2.1),
		yrange=(0.0, 1.1), 
		legend=false, 
		grid=false, 
		tick_dir=:in,
		yformatter=(a)->"", xformatter=(a)->"", 
		yticks=false)
	
	scatter!(plt, 1:x_top+1, cw_obs, color=color_scheme[2])

	plt2 = bar(
		1:x_top+1,
		(x)->f_γ(t, x),
		bar_width=0.04,
		xrange=(0, x_top+2.1),
		yrange=(0.0, 1.1), 
		# lw=3,
		legend=false, 
		grid=false, 
		axis=false,
		yformatter=(a)->"", xformatter=(a)->"", 
		yticks=false, 
		color=:black)

	scatter!(plt2, 1:x_top+1, (x)->f_γ(t, x), color=color_scheme[1])

	plt_both = plot(plt2, plt, layout=(2,1))


	plt = bar(
		1:x_top+1,
		(x)->f_γ(t, x) * cw_obs[x], 
		bar_width=0.04, 
		xrange=(0, x_top+2.1), 
		yrange=(0.0, 1.1), 
		legend=false, 
		grid=false, 
		tick_dir=:in,
		yformatter=(a)->"", xformatter=(a)->"", 
		yticks=false)
	
	scatter!(plt, 
			 1:x_top+1, 
			 (x)->f_γ(t, x) * cw_obs[x], 
			 color=color_scheme[5])

	plt_both, plt

end

# ╔═╡ 35892160-d3cc-4bb1-91f5-33ca27a1e488
let
	n = 4
	γ = 0.9
	t = 1
	h = Int(round(1/(1-γ)))
	plts = plot_cycleworld_gamma_arb_mult(t; T=50) do t, x
		# x < t+1 ? 0.0 : 0.9 ^ (x - (t+1))
		γⁿ_closed_form(γ, n, x-t) / (n == 1 ? 1.0 : max(γⁿ_closed_form(γ, n, (n-1)*h), γⁿ_closed_form(γ, n, (n-1)*h+1)))
	end
	plot(plts[1], plts[2], size=(700, 300))
end

# ╔═╡ 192b4b4e-8992-47fa-8fad-be98adeb0d4b


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
	
	@progress for step in 1:num_steps
		
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

# ╔═╡ ffd72f02-6104-4831-b272-f729c6c91c0b
cw_myopic_hrd, cw_myopic_p = let	
	env_size = 10
	num_steps = 100000
	lu = TDλ(0.1, 0.9)
	γ = 0.0
	horde, p = cycleworld_experiment(env_size, num_steps, TDλ(0.1, 0.9)) do 
		[[GVF(env_size, 
			FeatureCumulant(1), 
			OnPolicy(), 
			ConstantDiscount(γ))];
		[GVF(env_size, 
			PredictionCumulant(i), 
			OnPolicy(), 
			ConstantDiscount(γ)) for i in 1:9]]
	end
	horde, p
end

# ╔═╡ bb28674b-442e-4c84-bb5e-ba86b8d3c9db
let
	plotly()
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

# ╔═╡ 550e9a33-00d0-4312-849c-6b9c8d49e8c6
md"""
## Constant Discount
"""

# ╔═╡ 589b2c19-9b71-4a33-934e-c03b6fba851b
cw_cont_hrd, cw_cont_p = let	
	env_size = 10
	num_steps = 100000
	lu = TDλ(0.1, 0.9)
	γ = 0.9
	horde, p = cycleworld_experiment(env_size, num_steps, TDλ(0.1, 0.9)) do 
		[[GVF(env_size, 
			FeatureCumulant(1), 
			OnPolicy(), 
			ConstantDiscount(γ))];
		[GVF(env_size, 
			PredictionCumulant(i), 
			OnPolicy(), 
			ConstantDiscount(γ)) for i in 1:9]]
	end
	horde, p
end

# ╔═╡ c0c69cbc-6205-4b9f-93b0-86f2c65e226b
let
	plotly()
	horde = cw_cont_hrd
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
	# savefig(plt, "constant_discount.pdf")
	plt
end

# ╔═╡ d65e55ed-483f-436f-a4fa-091a2fea7a07


# ╔═╡ 54f0b409-c1be-4766-b63c-5b0c97e0b03b
# transformed_mso_term = let
# 	# norm_data_mso = unit_norm(data_mso)
# 	ret = [norm_data_mso]
# 	for i in 1:10
# 		vs = montecarlo_transform_iterate(ret[end]) do x, idx
# 			x > 0.9 ? 0.0 : 0.9
# 		end
# 		push!(ret, unit_norm(vs))
# 	end
# 	ret
# end

# ╔═╡ 5a116ab2-537e-4eb4-93ff-788ddf741fdf
md"""
## TerminatingDiscount
"""

# ╔═╡ f0eb7e3a-9f63-44de-8910-64669f985d09
cw_term_hrd, cw_term_p = let	
	env_size = 10
	num_steps = 100000
	lu = TDλ(0.1, 0.9)
	γ = 0.9
	horde, p = cycleworld_experiment(env_size, num_steps, TDλ(0.1, 0.9)) do 
		[[GVF(env_size, 
			FeatureCumulant(1), 
			OnPolicy(), 
			TerminatingDiscount(γ, 1))];
		[GVF(env_size, 
			PredictionCumulant(i), 
			OnPolicy(), 
			TerminatingDiscount(γ, 1)) for i in 1:9]]
	end
	horde, p
end

# ╔═╡ ea22ebd1-04b4-440f-b167-3086e0b445ad
let
	plotly()
	horde = cw_term_hrd
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
			color=color_scheme[4]; kwargs...)
	end
	plts = [plt]
	plts = [plts; [p_plot(i, xformatter=(a)->"") for i in 1:length(horde)-1]]
	
	plt_end = p_plot(length(horde), xtickfontsize=15)
	plt = plot(plts[1], plts[2], plts[3], plts[4], plt_end, layout = (5, 1))
	# savefig(plt, "term_discount.pdf")
	plt
end

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

# ╔═╡ acf34d2f-5219-46f7-9c3a-34d716131e5b


# ╔═╡ 24db8a0b-ea04-437d-af63-02709f41d357
md"""
### w/ Threshold Cumulant
"""

# ╔═╡ ed171485-39cf-4bba-8a42-12aafc4e6f92
cw_thrsh_hrd, cw_thrsh_p = let	
	env_size = 10
	num_steps = 100000
	lu = TDλ(0.1, 0.9)
	γ = 0.9
	horde, p = cycleworld_experiment(env_size, num_steps, TDλ(0.1, 0.9)) do 
		[[GVF(env_size, 
			FeatureCumulant(1), 
			OnPolicy(), 
			TerminatingDiscount(γ, 1))];
		[GVF(env_size, 
			ThresholdCumulant(PredictionCumulant(i), 0.5), 
			OnPolicy(), 
			TerminatingDiscount(γ, 1)) for i in 1:10]]
	end
	horde, p
end

# ╔═╡ af197fd3-f997-404e-acda-d8de0bba202d
let
	gr()
	horde = cw_thrsh_hrd
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
			color=color_scheme[6]; kwargs...)
	end
	plts = [plt]
	plts = [plts; [p_plot(i, xformatter=(a)->"") for i in 1:length(horde)-1]]
	
	plt_end = p_plot(length(horde), xtickfontsize=15)
	plt = plot(plts[1], plts[2], plts[3], plts[4], plt_end, layout = (5, 1))
	savefig(plt, "thrsh_cumulant.pdf")
	plt
end

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

# ╔═╡ dd22f696-0e8d-4e17-9422-b3c0b5e09ea2
md"# Random Intervals"

# ╔═╡ b3b5aff4-d0fb-4161-abad-f2f309db08d2
transformed_random_const = let
	data = random_data
	ret = [data]
	for i in 1:10
		vs = montecarlo_transform_iterate(ret[end]) do x, idx
			0.9
		end
		push!(ret, unit_norm(vs))
	end
	ret
end

# ╔═╡ 9840e05e-6f78-45e2-abcc-81f51dcc97e7
transformed_random_term = let
	data = random_data
	ret = [data]
	for i in 1:10
		vs = montecarlo_transform_iterate(ret[end]) do x, idx
			data[idx] > 0.9 ? 0.0 : 0.9
		end
		push!(ret, unit_norm(vs))
	end
	ret
end

# ╔═╡ 884ccf9b-f2d8-416c-8132-0dae04bacba3
transformed_random_term_v2 = let
	data = random_data
	ret = [data]
	for i in 1:10
		vs = montecarlo_transform_iterate(ret[end]) do x, idx
			x > 0.9 ? 0.0 : 0.9
		end
		push!(ret, unit_norm(vs))
	end
	ret
end

# ╔═╡ 95c58fbf-c05f-4f95-bf2d-28137ca04d4f
let
	
	gr()	
	yrng = 1000:2500
	d = transformed_random_term
	p_plot(x, y, y_term=nothing; kwargs...) = begin
		plt = plot(x, y;
			legend=nothing, 
			lw=2, 
			grid=false, 
			tick_dir=:out, 
			yformatter=(a)->"",
			yticks=false, 
			color=color_scheme[4], kwargs...)
		# plot!(yrng, (x)->0.9)
		if !isnothing(y_term) && sum(y_term) > 0
			# @info sum(y_term)
			yrng_c = collect(yrng)
			vline!(plt, yrng_c[y_term], color=:gray, linealpha=0.3; kwargs...)
		end
		plt
	end
	# d = transformed_mso_term

	γ_term(x) = x > 0.9
	d_term = [γ_term.(d[1]) for d_n in d]
	
	plt_first = p_plot(yrng, d[1][yrng], xformatter=(i)->"", color=color_scheme[2])
	plts = [p_plot(yrng, d[i][yrng], d_term[i-1][yrng]; xformatter=(i)->"")
				for i in 2:length(d)-1]
	plt_end = p_plot(yrng, d[end][yrng], d_term[end-1][yrng], xtickfontsize=15)
	plt = plot(plt_first, plts[1], plts[2], plts[3], plt_end, layouts=(5, 1))
	# plt = plot(plt_first, plts..., plt_end, layouts=(length(d), 1), size=(400, 1200))
	savefig(plt, "random_term.pdf")
	plt
end

# ╔═╡ 84109119-2233-4cd7-8e50-0cc1c69deef7
let
	
	gr()	
	yrng = 1000:2500
	d = transformed_random_term_v2
	p_plot(x, y, y_term=nothing; kwargs...) = begin
		plt = plot(x, y;
			legend=nothing, 
			lw=2, 
			grid=false, 
			tick_dir=:out, 
			yformatter=(a)->"",
			yticks=false, 
			color=color_scheme[4], kwargs...)
		plot!(yrng, (x)->0.9)
		if !isnothing(y_term) && sum(y_term) > 0
			# @info sum(y_term)
			yrng_c = collect(yrng)
			vline!(plt, yrng_c[y_term], color=:gray, linealpha=0.3; kwargs...)
		end
		plt
	end
	# d = transformed_mso_term

	γ_term(x) = x > 0.9
	d_term = [γ_term.(d_n) for d_n in d]
	
	plt_first = p_plot(yrng, d[1][yrng], xformatter=(i)->"", color=color_scheme[2])
	plts = [p_plot(yrng, d[i][yrng], d_term[i-1][yrng]; xformatter=(i)->"")
				for i in 2:length(d)-1]
	plt_end = p_plot(yrng, d[end][yrng], d_term[end-1][yrng], xtickfontsize=15)
	plt = plot(plt_first, plts[1], plts[2], plts[end-1], plt_end, layouts=(5, 1))
	# plt = plot(plt_first, plts..., plt_end, layouts=(length(d), 1), size=(400, 1200))
	# savefig(plt, "mso_term.pdf")
	plt
end

# ╔═╡ e2e6ca8f-1151-4226-bdf8-d9826500019e
let
	
	gr()	
	yrng = 1000:2500
	d = transformed_random_const
	p_plot(x, y, y_term=nothing; kwargs...) = begin
		plt = plot(x, y;
			legend=nothing, 
			lw=2, 
			grid=false, 
			tick_dir=:out, 
			yformatter=(a)->"",
			yticks=false, 
			color=color_scheme[1], kwargs...)
		# plot!(yrng, (x)->0.9)
		plt
	end
	# d = transformed_mso_term


	
	plt_first = p_plot(yrng, d[1][yrng], xformatter=(i)->"", color=color_scheme[2])
	plts = [p_plot(yrng, d[i][yrng]; xformatter=(i)->"")
				for i in 2:length(d)-1]
	plt_end = p_plot(yrng, d[end][yrng], xtickfontsize=15)
	plt = plot(plt_first, plts[1], plts[2], plts[3], plt_end, layouts=(5, 1))
	# plt = plot(plt_first, plts[2], plts[3], plt_end, layouts=(length(d), 1), size=(400, 1200))
	# savefig(plt, "mso_term.pdf")
	savefig(plt, "random_const.pdf")
	plt
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

# ╔═╡ d3f3506d-b190-4d73-baa9-41a80bd60e2c
transformed_mso_term = let
	norm_data_mso = unit_norm(data_mso)
	ret = [norm_data_mso]
	for i in 1:10
		vs = montecarlo_transform_iterate(ret[end]) do x, idx
			x > 0.9 ? 0.0 : 0.9
		end
		push!(ret, unit_norm(vs))
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

# ╔═╡ bae2112a-265c-4620-a51e-6cf2b877ab72
let
	gr()
	yrng = 100:1000
	p_plot(x, y; kwargs...) = begin
		plot(x, y,
			legend=nothing, 
			lw=2, 
			grid=false,
			tick_dir=:out, 
			yformatter=(a)->"",
			yticks=false, 
			color=color_scheme[1]; kwargs...)
	end
	d = transformed_mso
	plt_first = p_plot(yrng, unit_norm(d[1][yrng]), xformatter=(i)->"", color=color_scheme[2])
	plts = [p_plot(yrng, unit_norm(tmso[yrng]), xformatter=(i)->"")
				for tmso in d[1:end-1]]
	plt_end = p_plot(yrng, unit_norm(d[end][yrng]), xtickfontsize=15)
	plt = plot(plt_first, plts[2], plts[3], plts[4], plt_end, layouts=(5, 1))
	savefig(plt, "mso_constant.pdf")
	plt
end

# ╔═╡ 57045457-fddd-4cdc-bdb2-92c0d2cdb88d
let
	
	gr()	
	yrng = 100:1000
	p_plot(x, y, y_term=nothing; kwargs...) = begin
		plt = plot(x, y;
			legend=nothing, 
			lw=2, 
			grid=false, 
			tick_dir=:out, 
			yformatter=(a)->"",
			yticks=false, 
			color=color_scheme[4], kwargs...)

		if !isnothing(y_term)
			yrng_c = collect(yrng)
			vline!(plt, yrng_c[y_term], color=:gray, linealpha=0.3; kwargs...)
		end
		plt
	end
	d = transformed_mso_term

	γ_term(x) = x > 0.9
	d_term = [γ_term.(d_n) for d_n in d]
	
	plt_first = p_plot(yrng, d[1][yrng], xformatter=(i)->"", color=color_scheme[2])
	plts = [p_plot(yrng, d[i][yrng], d_term[i-1][yrng]; xformatter=(i)->"")
				for i in 2:length(d)-1]
	plt_end = p_plot(yrng, d[end][yrng], d_term[end-1][yrng], xtickfontsize=15)
	plt = plot(plt_first, plts[1], plts[2], plts[3], plt_end, layouts=(5, 1))
	savefig(plt, "mso_term.pdf")
	plt
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
			  butterworth_filter(2, ret[end], cutoff_freq=0.09))
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
	plot(transformed_mso_bw[10][yrng], label="butterworth", lw = 2)
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
	plt = plot(sine_analysis_mc[0.0][1][yrng], palette=color_scheme)
	for γ in sw_gammas[1:end-2]
		plot!(plt, sine_analysis_mc[γ][2][yrng], label=γ, palette=color_scheme)
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

# ╔═╡ 942801a0-dbf4-46fa-8baf-6bf72eb98cfe
md"""
# Critterbot Experiments
"""

# ╔═╡ 15a67f1f-e8f3-49bf-911f-44efdf16f4ab
names(critterbot_data)

# ╔═╡ 1250743e-7e38-4342-b3b7-e89f5b5fe23e
@bind sensor_name PlutoUI.Select(names(critterbot_data))

# ╔═╡ 8b0c504c-d1ad-494f-978d-be2521e3e975
plot(critterbot_data[!, Symbol(sensor_name)])

# ╔═╡ 2adfe1a3-1419-470d-a7a7-63f6c82b86b7
cb_mc = let
	d = Dict{Float64, Vector{Vector{Float64}}}()
	for γ ∈ [0.9]
		norm_data_mso = unit_norm(critterbot_data[!, :Light3])
		ret = [norm_data_mso]
		@progress for i in 1:100
			push!(ret, montecarlo_transform_iterate(γ, ret[end]))
		end
		d[γ] = ret
	end
	d
end

# ╔═╡ 97c5c775-2cbf-4ca3-bec3-40ccb18b3082
let
	plotly()
	yrng = 2000:3000
	plt = plot(unit_norm(cb_mc[0.9][1])[yrng])
	for i in 2:10
		plot!(plt, unit_norm(cb_mc[0.9][i])[yrng], label=i)
	end
	plt
end

# ╔═╡ 92a3653d-5b6a-446b-a6cd-110ae1bde679
let
	gr()
	# yrng = [6500:12500
	# yrng = [7000:8200; 11600:12500]
	yrng = [7000:8100; 11000:12500]
	p_plot(y; kwargs...) = begin
		plot(y,
			legend=nothing, 
			lw=2, 
			grid=false, 
			tick_dir=:out, 
			yformatter=(a)->"",
			yticks=false, 
			color=color_scheme[1]; kwargs...)
	end
	first_plt = p_plot(unit_norm(cb_mc[0.9][1])[yrng], xformatter=(a)->"", color=color_scheme[2])
	plts = [p_plot(unit_norm(cb_mc[0.9][i])[yrng], xformatter=(a)->"") for i in 2:39]
	last_plt = p_plot(unit_norm(cb_mc[0.9][40])[yrng], xtickfontsize=15)
	plt = plot(first_plt, plts[6], plts[11], plts[21], last_plt, layout=(5, 1)) 
	savefig(plt, "cb_const.pdf")
	plt
end

# ╔═╡ f493245f-33df-4705-90f6-7f8d3398733a
@info length(cb_mc[0.9])

# ╔═╡ 3132737d-aa1d-47e8-8e11-cddc44be039c
cb_mc_term = let
	norm_data_mso = unit_norm(critterbot_data[!, :Light3])
	ret = [norm_data_mso]
	@progress for i in 1:30
		vs = montecarlo_transform_iterate(ret[end]) do x, t
			norm_data_mso[t] > 0.99 ? 0.0 : 0.9
		end
		push!(ret, unit_norm(vs))
	end
	ret	
end

# ╔═╡ e7b95a71-780f-48dd-99b6-f9ef863dae19
let
	gr()
	# yrng = 6500:15000
	# yrng = [7000:8500; 11000:12500]
	
	p_plot(y, y_term=nothing; kwargs...) = begin
		plt = plot(y,
			legend=nothing, 
			lw=2, 
			grid=false, 
			tick_dir=:out, 
			yformatter=(a)->"",
			yticks=false, 
			color=color_scheme[4]; kwargs...)

		if !isnothing(y_term) && any(y_term)
			yrng_c = collect(yrng)
			# @info size(y_term)
			term_idx = findall(y_term)
			vline!(plt, term_idx, color=:gray, linealpha=0.3; kwargs...)
		end
		plt
	end

	yrng = [7000:8100; 11000:12500]
	d = cb_mc_term

	γ_term(x) = x > 0.9
	d_term = [γ_term.(d_n) for d_n in d]
	
	plt_first = p_plot(d[1][yrng], xformatter=(i)->"", color=color_scheme[2])
	plts = [p_plot(d[i][yrng], d_term[1][yrng], xformatter=(i)->"")
				for i in 2:length(d)-1]
	plt_end = p_plot(d[end][yrng], d_term[1][yrng], xtickfontsize=15)
	plt = plot(plt_first, plts[5], plts[10], plts[20], plt_end, layouts=(5, 1))

	savefig(plt, "cb_term.pdf")
	plt
end

# ╔═╡ bac4e97b-acd8-409f-bb51-3ea470fc33ed
let
	d = cb_mc_term
	γ_term(x) = x > 0.9
	# d_term = [γ_term.(d_n) for d_n in d]
	yrng = [7000:8100; 11000:12500]
	# d_term[1][yrng]
	findall((x)->γ_term(x), d[1][yrng])
end

# ╔═╡ ddd34a5a-38eb-4f89-a515-093f1d0275cb
cb_bw = let
	
	norm_data_mso = unit_norm(critterbot_data[!, :Light3])
	ret = [norm_data_mso]
	for i in 1:10
		push!(ret, 
			  butterworth_filter(1, ret[end], cutoff_freq=0.8))
	end
	ret
end

# ╔═╡ df66901c-074d-42a7-91e5-db5e81048881
let
	yrng = 1100:1500
	plt = plot(unit_norm(cb_bw[1])[yrng])
	for i in 2:length(cb_bw)
		plot!(plt, cb_bw[i][yrng], label=i)
	end
	plt
end

# ╔═╡ 43b85260-18ed-4035-b1c1-86946b7e89fd
let
	yrng = 1100:1500
	plt = plot(unit_norm(cb_mc[0.9][1])[yrng], palette=color_scheme)
	for i in 2:10
		plot!(plt, unit_norm(cb_mc[0.9][i])[yrng], label=i, palette=color_scheme)
	end
	plt
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
FourierAnalysis = "e7e9c730-dc46-11e9-3633-f1ab55cc17e1"
HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
MinimalRLCore = "4557a151-568a-41c4-844f-9d8069264cea"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Revise = "295af30f-e4ad-537b-8983-00126c2a3abe"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
DSP = "~0.7.4"
DataFrames = "~1.3.2"
FourierAnalysis = "~1.2.0"
HDF5 = "~0.16.2"
MinimalRLCore = "~0.2.1"
Plots = "~1.23.6"
PlutoUI = "~0.7.35"
ProgressLogging = "~0.1.4"
Revise = "~3.3.2"
StatsPlots = "~0.14.30"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "7e9cc0126e4ec82321a6be7ddad6d54ae3b92901"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "16b6dbc4cf7caee4e1e75c49485ec67b667098a0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.3.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "cc37d689f599e8df4f464b2fa3870ff7db7492ef"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

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

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

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

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "64df3da1d2a26f4de23871cd1b6482bb68092bd5"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.3"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "4f619d394ac521dc59cb80a2cd8f78578e483a9d"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.2.1"

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
git-tree-sha1 = "7a60c856b9fa189eb34f5f8a6f6b5529b7942957"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

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
git-tree-sha1 = "da8b06f89fce9996443010ef92572b193f8dca1f"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.7.8"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "db2a9cb664fcea7836da4b414c3278d71dd602d2"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.6"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

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

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "49eba9ad9f7ead780bfb7ee319f962c811c6d3b2"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.8"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "da9e1a9058f8d3eec3a8c9fe4faacfb89180066b"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.86"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.Extents]]
git-tree-sha1 = "5e1e4c53fa39afe63a7d356e30452249365fba99"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.1"

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

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "f9818144ce7c8c41edf5c4c179c684d92aa4d9fe"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.6.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "d3ba08ab64bdfd27234d3f61956c966266757fe6"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.7"

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

[[deps.FourierAnalysis]]
deps = ["AbstractFFTs", "DSP", "FFTW", "LinearAlgebra", "PosDefManifold", "RecipesBase", "Statistics"]
git-tree-sha1 = "f7dc4f9754ac1c6590133004dedb1fbda5a07f1e"
uuid = "e7e9c730-dc46-11e9-3633-f1ab55cc17e1"
version = "1.2.2"

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
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "1cd7f0af1aa58abc02ea1d872953a97359cb87fa"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.4"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "30f2b340c2fff8410d89bfcdc9c0a6dd661ac5f7"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.62.1"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "bc9f7725571ddb4ab2c4bc74fa397c1c5ad08943"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.69.1+0"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "e07a1b98ed72e3cdd02c6ceaab94b8a606faca40"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.2.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "fe9aea4ed3ec6afdfbeb5a4f39a2208909b162a6"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.5"

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

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "Mmap", "Random", "Requires", "UUIDs"]
git-tree-sha1 = "3dab31542b3da9f25a6a1d11159d4af8fdce7d67"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.16.14"

[[deps.HDF5_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "OpenSSL_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "4cc2bb72df6ff40b055295fdef6d92955f9dede8"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.12.2+2"

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

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

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
git-tree-sha1 = "b7bc05649af456efc75d178846f47006c2c4c3c7"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.6"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.InvertedIndices]]
git-tree-sha1 = "82aec7a3dd64f4d9584659dc0b62ef7db2ef3e19"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.2.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

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
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "d9ae7a9081d9b1a3b2a5c1d3dac5e2fdaafbd538"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.22"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "9816b296736292a80b9a3200eb7fbb57aaa3917a"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.5"

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

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

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

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "60168780555f3e663c536500aa790b6368adc02a"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.3.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

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

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "7008a3412d823e29d370ddc77411d593bd8a3d03"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.9.1"

[[deps.NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "2c3726ceb3388917602169bed973dbc97f1b51a8"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.13"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "82d7c9e310fe55aa54996e6f7f94674e2a38fcb4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.9"

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

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "478ac6c952fddd4399e71d4779797c538d0ff2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.8"

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
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "c95373e73290cf50a8a22c3375e4625ded5c5280"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun"]
git-tree-sha1 = "0d185e8c33401084cab546a756b387b15f76720c"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.23.6"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "5bb5129fdd62a2bbbe17c2756932259acf467386"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.50"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase"]
git-tree-sha1 = "a14a99e430e42a105c898fcc7f212334bc7be887"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "3.2.4"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PosDefManifold]]
deps = ["LinearAlgebra", "Statistics"]
git-tree-sha1 = "54ada42dbaca60ecc4ffcf0c5db02de37e6ee520"
uuid = "f45a3650-5c51-11e9-1e9a-133aa5e309cf"
version = "0.5.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

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

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "786efa36b7eff813723c4849c90456609cf06661"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "261dddd3b862bd2c940cf6ca4d1c8fe593e457c8"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.3"

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

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "c73149ff75d4efb19b6d77411d293ae8fb55c58e"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.3.4"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "77d3c4726515dca71f6d80fbb5e251088defe305"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.18"

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

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "2d7d9e1ddadc8407ffd460e24218e37ef52dd9a3"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.16"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "8d7530a38dbd2c397be7ddd01a424e4f411dcc41"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.2"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "4d9c69d65f1b270ad092de0abe13e859b8c55cad"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.33"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "b03a3b745aa49b566f128977a7dd1be8711c5e71"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.14"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

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
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "c79322d36826aa2f4fd8ecfa96ddb47b174ac78d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.0"

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

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

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
# ╟─a0c15db4-d43f-40ce-ab35-e7583b7c0dd4
# ╟─729faad8-176f-4b6b-af97-3e321c7be3e0
# ╠═ebe424a8-f385-46ee-80e9-12b99bb2e691
# ╠═098f4309-d067-4d4e-be6c-8dacdc059b89
# ╠═44ccdaa4-d952-4367-8bf0-15c8841a253d
# ╠═b4d08bb8-ea59-11eb-1df0-93d1149fcf0a
# ╠═5590501e-eab5-4d08-a81f-b1f5a2b91a08
# ╠═1df5d910-1a95-4049-906f-770eb3a7990a
# ╠═223375db-94c7-4ac1-bdae-0d6856e2f492
# ╠═24457b9c-64a9-43a4-b3c1-b1f5af7ed823
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
# ╠═3cbe38c5-11a8-4d99-b5b7-613eced6139b
# ╠═512344a1-e3ba-4a6f-ab87-b5ef344c413d
# ╠═2c639be3-3867-45e2-80f3-3c0890529f1c
# ╠═ba06366c-13c7-4112-a8f9-998346c49910
# ╠═5ba27c96-e3a1-4ebd-b210-72bf4262cdfb
# ╠═4bf138a4-d5cd-401c-a627-4633b224f4d7
# ╠═fc965f74-158c-445f-b822-b00840e5de1b
# ╠═37cf17e2-7796-4342-8ca6-cd8bf9a4ed9b
# ╠═9192754f-a710-4f8e-8c0b-a11de74bc240
# ╠═e2eeb4ec-06d4-4731-ad25-b9c956457d11
# ╠═c8b3fe86-4433-4d08-b303-8cd9d91f9953
# ╠═52bb70f3-6309-41c6-a65c-6720db2b4a05
# ╟─35ed964f-819e-4f46-a531-3da2689356f4
# ╠═792c90c4-f66c-4695-8897-f7bb990ca31e
# ╠═e7c38a2a-343a-48dc-bce4-0fe5df04c65d
# ╠═fbe6c146-87d2-4ee6-b9b9-1818e80e6959
# ╠═e4deba4c-f5a9-4736-b95c-2ef51e8e8f3e
# ╠═69b43c3e-e180-449d-a1fe-91d686c183e8
# ╠═79ebfb96-5bb2-4337-839e-bc1ac81cb0a7
# ╠═74d3635c-304d-40f6-a1b4-f8911411d933
# ╠═650de732-e312-41d5-938c-beb5e330ee0f
# ╠═c88e09f3-88c4-4e27-837c-bce455447e99
# ╠═94eaa0f6-cbd6-4d10-a24c-f6296048633b
# ╠═dd1c2f72-379a-4013-a5fc-adf717905b74
# ╠═5a79876d-d7bf-4829-9658-4ea4477b0349
# ╠═d3ed34bb-3f5d-4565-8465-dc1fa0cc1f9b
# ╟─1568abbd-856f-46bb-bf30-86ee3e6553c6
# ╟─91957826-379f-4db4-9a61-fc7c5fa667b1
# ╠═931db888-1f65-4ac8-965a-e3fa12672ea4
# ╠═62fde1ef-c6db-4ba7-b484-e8c9ac530f69
# ╠═3a4d265e-d226-480f-ba8d-6064f601c6f1
# ╠═e7643fc3-f032-47cb-b717-97516b7a696b
# ╟─611e00d5-cf92-4e0b-9504-8b611b88a897
# ╟─46c3ba82-c56e-4ee1-9fad-2523137ba84d
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
# ╠═d5cc227c-f5f7-47aa-9027-597879fc02df
# ╠═eaeb04c5-5371-4811-849e-e3805768d1c2
# ╠═99e4a6e9-caa4-436a-b05d-8059889ca95e
# ╠═ad61416a-3c81-44a9-a2b1-26879b9499c7
# ╠═6a63a5b5-f42f-4e97-9d43-eced19c01cf9
# ╠═51255b81-9bab-4614-a6fe-22c9db4d40d8
# ╠═ae6c39ee-2366-41d8-a576-b43e1c441d6b
# ╠═f20bd192-bc92-4a74-8c24-50b32f83b604
# ╠═8111936a-2c3d-4f2c-8f6c-e25feb60ec72
# ╠═04be37d5-9934-481b-98e1-da94dba09e87
# ╟─a88b0b03-f496-4f70-8271-5a371d0ce769
# ╠═8049b4cd-459c-4ae1-8d28-b357d073a527
# ╠═35892160-d3cc-4bb1-91f5-33ca27a1e488
# ╠═192b4b4e-8992-47fa-8fad-be98adeb0d4b
# ╟─64a1c059-6dc7-44b5-8cef-f5d517871aab
# ╟─1dd3e976-d74b-40bf-ab23-642fc6ccd5ea
# ╠═16671204-227f-436c-9e1d-f4f5738df3f9
# ╠═c06a9fca-e4bc-4748-9970-268786ee1f5a
# ╠═7c77be27-abdf-4625-951d-2601cbac7d84
# ╟─18765352-0a28-44b5-a8a3-fa1063e84da3
# ╠═ffd72f02-6104-4831-b272-f729c6c91c0b
# ╠═bb28674b-442e-4c84-bb5e-ba86b8d3c9db
# ╟─550e9a33-00d0-4312-849c-6b9c8d49e8c6
# ╠═589b2c19-9b71-4a33-934e-c03b6fba851b
# ╠═c0c69cbc-6205-4b9f-93b0-86f2c65e226b
# ╠═d65e55ed-483f-436f-a4fa-091a2fea7a07
# ╠═54f0b409-c1be-4766-b63c-5b0c97e0b03b
# ╟─5a116ab2-537e-4eb4-93ff-788ddf741fdf
# ╠═f0eb7e3a-9f63-44de-8910-64669f985d09
# ╠═ea22ebd1-04b4-440f-b167-3086e0b445ad
# ╠═e82378ac-e02a-4425-ab53-a372fa831a40
# ╠═acf34d2f-5219-46f7-9c3a-34d716131e5b
# ╟─24db8a0b-ea04-437d-af63-02709f41d357
# ╠═ed171485-39cf-4bba-8a42-12aafc4e6f92
# ╠═af197fd3-f997-404e-acda-d8de0bba202d
# ╠═04cbe365-e019-4963-a191-68ff02fd13b3
# ╠═dd22f696-0e8d-4e17-9422-b3c0b5e09ea2
# ╠═b3b5aff4-d0fb-4161-abad-f2f309db08d2
# ╠═9840e05e-6f78-45e2-abcc-81f51dcc97e7
# ╠═884ccf9b-f2d8-416c-8132-0dae04bacba3
# ╠═95c58fbf-c05f-4f95-bf2d-28137ca04d4f
# ╠═84109119-2233-4cd7-8e50-0cc1c69deef7
# ╠═e2e6ca8f-1151-4226-bdf8-d9826500019e
# ╟─f56773f8-57aa-4157-bc65-dea6bce7f6cc
# ╟─77073f41-cde0-42fb-a4b9-9a6ef3285923
# ╠═04e159db-2740-41ae-b543-8e4f0874fb3b
# ╠═d63b2595-c84d-4b14-88b9-b783896655ef
# ╠═d3f3506d-b190-4d73-baa9-41a80bd60e2c
# ╠═cb691a4f-c0cd-490e-9339-65a2c3997071
# ╠═f6b19b92-ce33-41a0-a8b1-d239a94604d1
# ╠═162cad05-d12a-482b-b3a0-dad61d494b5b
# ╠═bae2112a-265c-4620-a51e-6cf2b877ab72
# ╠═57045457-fddd-4cdc-bdb2-92c0d2cdb88d
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
# ╠═942801a0-dbf4-46fa-8baf-6bf72eb98cfe
# ╠═15a67f1f-e8f3-49bf-911f-44efdf16f4ab
# ╠═1250743e-7e38-4342-b3b7-e89f5b5fe23e
# ╠═8b0c504c-d1ad-494f-978d-be2521e3e975
# ╠═2adfe1a3-1419-470d-a7a7-63f6c82b86b7
# ╠═97c5c775-2cbf-4ca3-bec3-40ccb18b3082
# ╠═92a3653d-5b6a-446b-a6cd-110ae1bde679
# ╠═f493245f-33df-4705-90f6-7f8d3398733a
# ╠═3132737d-aa1d-47e8-8e11-cddc44be039c
# ╠═e7b95a71-780f-48dd-99b6-f9ef863dae19
# ╠═bac4e97b-acd8-409f-bb51-3ea470fc33ed
# ╠═ddd34a5a-38eb-4f89-a515-093f1d0275cb
# ╠═df66901c-074d-42a7-91e5-db5e81048881
# ╠═43b85260-18ed-4035-b1c1-86946b7e89fd
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
