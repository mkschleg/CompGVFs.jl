### A Pluto.jl notebook ###
# v0.18.1

using Markdown
using InteractiveUtils

# ╔═╡ 4c5ea1f7-ddd5-44fe-be88-b582803dbe75
using Random, Revise

# ╔═╡ 330cb40c-0ea1-41c5-bed3-c3ade059555b
using DSP

# ╔═╡ fc6e55f1-60a9-4c82-892e-9555f2f70ba1
using PlutoUI, StatsPlots, Plots

# ╔═╡ 26c6d74e-8c06-4d60-b52f-cbe8b05ecbee
using ProgressLogging

# ╔═╡ bf2b7016-167b-4c8d-8616-a5b8847223f3
using MinimalRLCore

# ╔═╡ ca3a10fa-511f-46ea-b71a-40aae51e7f34
md"""
# Introduction

## This notebook
"""

# ╔═╡ b927266d-01b0-4fb0-b8e2-03a351b4bb9f
md"""
# Data
"""

# ╔═╡ 93585f0c-256a-43a6-9c54-8d39791b9b44
md"""
## Cycle World
"""

# ╔═╡ f31e525c-85be-4c9f-8388-5f50790d9764
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

# ╔═╡ cebeb2d9-e1b6-467e-ac54-32f405f5715e
function MinimalRLCore.reset!(env::CycleWorld, rng=nothing; kwargs...)
	env.agent_state = 0
end

# ╔═╡ 6cb07467-a793-4cf3-9afd-361eb487b579
MinimalRLCore.get_actions(env::CycleWorld) = Set(1:1)

# ╔═╡ a896d5ce-84f1-41c3-aa7f-45ea40da3689
function MinimalRLCore.environment_step!(
		env::CycleWorld,
		action, 
		rng=nothing; 
		kwargs...)
	env.agent_state = (env.agent_state + 1) % env.chain_length
end

# ╔═╡ b9f4121a-32a1-4d6d-b831-8e7ffedf8846
MinimalRLCore.get_reward(env::CycleWorld) = 0 # -> get the reward of the environment

# ╔═╡ eefb17c9-3e54-4cf4-9b98-73d7b73553d8
fully_observable_state(env::CycleWorld) = [env.agent_state+1]

# ╔═╡ 291f437e-ab7a-45d3-983d-89881b4a7267
function partially_observable_state(env::CycleWorld)
	state = zeros(1)
	if env.agent_state == 0
		state[1] = 1
	end
	return state
end

# ╔═╡ d0a4b432-550a-4536-87a0-42abbf66ada8
function partially_observable_state(state::Int)
	state = zeros(1)	
	if state == 0
		state[1] = 1
	end
	return state
end

# ╔═╡ 82757b09-e34d-4d38-94e4-f1bf34a7a242
function MinimalRLCore.get_state(env::CycleWorld) # -> get state of agent
	if env.partially_observable
		return partially_observable_state(env)
	else
		return fully_observable_state(env)
	end
end

# ╔═╡ 45ff69fc-9b2a-456f-b894-8522748c55a2
function MinimalRLCore.is_terminal(env::CycleWorld) # -> determines if the agent_state is terminal
	return false
end

# ╔═╡ 054aecaa-9371-42a0-96c6-1dc9d03f8950
function Base.show(io::IO, env::CycleWorld)
	model = fill("0", env.chain_length)
	model[1] = "1"
	println(io, join(model, ' '))
	model = fill("-", env.chain_length)
	model[env.agent_state + 1] = "^"
	println(io, join(model, ' '))
	# println(env.agent_state)
end

# ╔═╡ bc4b2237-f34d-4909-a5c1-0b296b0921f5
md"""
## MSO
"""

# ╔═╡ 2b2e79d0-01aa-4a55-9bc3-4f81dacae075
begin
	mutable struct MSO <: MinimalRLCore.AbstractEnvironment
		θ::Int
		Ω::Vector{Float64}
		state::Vector{Float64}
	end
	MSO() = MSO(1, [0.2, 0.311, 0.42, 0.51], [0.0])

end

# ╔═╡ ec59e1d0-4c36-42c2-a774-36723b425be0
function MinimalRLCore.step!(self::MSO)
	self.state[1] = sum([sin(self.θ*ω) for ω in self.Ω])
	self.θ += 1
	return self.state
end

# ╔═╡ a8db4a35-0fe5-4fee-b9d5-81b6a2dd43d2
function MinimalRLCore.start!(self::MSO)
	self.θ = 1
	return step!(self)
end

# ╔═╡ 95e3fab5-c896-414a-bb1f-88f59d86344a
MinimalRLCore.get_state(self::MSO) = self.state

# ╔═╡ cde0a299-2d15-4e67-90e5-51000f687852
get_num_features(self::MSO) = 1

# ╔═╡ b2f12306-6e74-41f4-b401-b4fdee723b09
get_num_targets(self::MSO) = 1

# ╔═╡ 29c34186-3232-4fc2-939e-4cb62cd62e12
md"""
## Critterbot
"""

# ╔═╡ be716093-89f3-4933-b992-c166e70239d5
import HDF5

# ╔═╡ 3848090d-3b2f-4e76-8ae6-05da81517e33
md"""
# GVF details
"""

# ╔═╡ 8c08b2d0-7d4d-4900-af08-c1572f01a149
md"""
## GVF
"""

# ╔═╡ 6fce8cae-55b6-4a36-b2bf-03a25634b38e
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

# ╔═╡ c5807b71-26fd-46d3-aedc-65809d687e38
predict(gvf::GVF{<:AbstractVector}, x::AbstractVector{<:Number}) = 
	dot(gvf.w, x)

# ╔═╡ 843c7fa8-8894-43e7-bd0b-20498727cc4f
predict(gvf::GVF{<:AbstractVector}, x::AbstractVector{Int}) = begin; 
	w = gvf.w; 
	ret = sum(view(w, x))
end

# ╔═╡ 9077cc84-3975-4494-95c1-90f0e11c9f2e
predict(gvf::GVF{<:AbstractVector}, x::Int) = begin; 
	gvf.w[x]
end

# ╔═╡ f7426601-f710-4d6d-a100-0f1e5c0fa1a4
md"""
## Parameter Functions
"""

# ╔═╡ ed399039-a185-4f8e-9ee0-eca1df7c1dfc
md"""
### Cumulants
"""

# ╔═╡ d53e0ee5-baef-4891-a720-c54681af163c
struct FeatureCumulant
    idx::Int
end

# ╔═╡ 3e493422-8c49-4075-9d91-2826f194a1bd
get_value(fc::FeatureCumulant, o, x, p, r) = x[fc.idx]

# ╔═╡ 19733780-8f1c-4934-b38b-fa876069478e
get_value(fc::FeatureCumulant, o, x::Vector{Int}, p, r) = fc.idx ∈ x

# ╔═╡ 2b8eaa2a-16f4-4bde-9df6-4a05ba2a445d
struct ObservationCumulant
    idx::Int
end

# ╔═╡ 6d20cf86-c4c7-4977-9db2-07bece906045
get_value(oc::ObservationCumulant, o, x, p, r) = o[oc.idx]

# ╔═╡ 7a913e09-6ce3-4452-a63e-551f768495ec
struct PredictionCumulant
	idx::Int
end

# ╔═╡ e1d9aa7b-f371-4ea9-8e5a-30d69fd20fce
get_value(pc::PredictionCumulant, o, x, p, r) = p[pc.idx]

# ╔═╡ c515c139-a8ec-4447-901d-0f525d0017d9
struct RescaleCumulant{C, F}
	c::C
	γ::F
end

# ╔═╡ 68ac9061-cdbd-41b1-9f4b-b352874952bb
get_value(rsc::RescaleCumulant, o, x, p, r) = 
	get_value(rsc.c, o, x, p, r)*(1-rsc.γ)

# ╔═╡ 17130de6-9f7d-4d0a-9e43-e8dbad2726af
struct ThresholdCumulant{C, F}
	c::C
	θ::F
end

# ╔═╡ d7fb81bd-970d-4091-85fc-48a364ca57ef
get_value(rsc::ThresholdCumulant, o, x, p, r) = 
	get_value(rsc.c, o, x, p, r) >= rsc.θ ? 1 : 0

# ╔═╡ 670db08e-c9af-4c18-a86f-d37ceef09f9c
md"""
### Policies
"""

# ╔═╡ b7a7b85e-d489-4a3a-b52f-5fec58737b0f
struct OnPolicy end

# ╔═╡ 53cd97e4-5ce8-4096-b8ba-b61880f80b57
get_value(op::OnPolicy, args...) = 1.0

# ╔═╡ d5326af3-e9e3-4b75-8329-4f074f9fb37d
md"""
### Discount
"""

# ╔═╡ 0064e10a-2c86-4840-ac27-044341321021
struct ConstantDiscount{F}
	γ::F
end

# ╔═╡ 840ab122-fae5-4870-993e-ab7119fa4cfe
get_value(cd::ConstantDiscount, args...) = cd.γ

# ╔═╡ 5b18e194-720c-4cb7-b47b-88147795bddb
struct TerminatingDiscount{F}
	γ::F
	idx::Int
end

# ╔═╡ 97fd8519-3033-49a5-bf19-81f982a5d6a3
get_value(fc::TerminatingDiscount, o, x::Vector{Int}) = fc.idx ∈ x ? zero(typeof(fc.γ)) : fc.γ

# ╔═╡ a8d0b5c3-6a81-4380-a06c-66a2baae8394
md"""
## Horde
"""

# ╔═╡ 9bb09cae-6051-44c3-996a-0ccb14902db2
const Horde = Vector{<:GVF}

# ╔═╡ 461e5bc1-bc76-4f79-8a28-5f95265d9e3d
predict(horde::Horde, x) = [predict(gvf, x) for gvf in horde]

# ╔═╡ 37b2ef37-2136-4363-8df2-967f7df12475
md"""
# Data Transforms
"""

# ╔═╡ c961895d-ba02-444a-85a6-9b2128c6c572
md"""
## TD(λ)
"""

# ╔═╡ 08c0950c-bd12-4ae8-b62d-057c67a5c83c
struct TDλ
    α::Float32
    λ::Float32
end

# ╔═╡ 66892139-2087-401d-83b3-a7205813b1f7
function update!(
		lu::TDλ, 
		gvf,
		x_t, x_tp1, 
		ρ_t, c, γ_t, γ_tp1)
    
    λ = lu.λ

	w = gvf.w
	z = gvf.z
	
    δ = c + γ_tp1*predict(gvf, x_tp1) - predict(gvf, x_t)
    
    if eltype(x_t) <: Integer # specialize for int features
        z .*= γ_t*λ
        view(z, x_t) .+= 1
        z .*= ρ_t
        w .+= (lu.α * δ) .* z
    else
        z .= ρ_t .* ((γ_t*λ) .* gvf.z .+ x_t)
        w .+= lu.α * δ * gvf.z
    end
end

# ╔═╡ 7a6a45ee-bcfc-4833-8955-3022d6d77210
function update!(lu, 
				 gvf::GVF, 
				 o_t, x_t, a_t, μ_t, 
				 o_tp1, x_tp1, r_tp1, p_tp1)
	
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

# ╔═╡ 4af3253d-5fba-4d4f-bd08-66d6c7e55347
function update!(lu, gvfs::Vector{G}, args...) where G<:GVF
    Threads.@threads for i in 1:length(gvfs)
		update!(lu, gvfs[i], args...)
    end
end

# ╔═╡ 9a1d6978-e078-44c3-aac4-7039f368d16b
md"""
## Monte Carlo
"""

# ╔═╡ d24de518-5d73-410f-89df-68668a2b2d99
function montecarlo_transform(γ, seq)
	# ret = zeros(eltype(seq), length(seq))
	s = zero(eltype(seq))
	for r in Iterators.reverse(seq)
		s = r + γ*s
	end
	s
end

# ╔═╡ af68d866-f508-4ea4-9597-f3378443a932
function montecarlo_transform_iterate(γ, seq)
	rets = zeros(eltype(seq), length(seq)+1)
	# for (r_idx, r) in Iterators.reverse(enumerate(seq))
	for t in length(seq)-1:-1:1
		rets[t] = seq[t+1] + γ*rets[t+1]
	end
	rets[1:end-1]
end

# ╔═╡ 3818803c-b121-4128-894b-44749d8fc214
function montecarlo_transform_iterate(γ::Function, seq)
	rets = zeros(eltype(seq), length(seq)+1)
	for t in length(seq)-1:-1:1
		rets[t] = seq[t+1] + γ(seq[t+1], t+1)*rets[t+1]
		# end
	end
	rets[1:end-1]
end

# ╔═╡ d57e9370-3ffb-4f3e-94e0-212968c2113d
let
	x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.6, 0.8, 0.9, 0.91, 0.91, 0.8, 0.5, 0.1, 0.0, 0.0]
	montecarlo_transform_iterate(x) do x, t
		x > 0.9 ? 0.0 : 0.9
	end
end

# ╔═╡ 6bee6c47-d0e9-4b39-9807-f1f90d8a98af
md"""
## Butterworth Filters
"""

# ╔═╡ 193fdee2-008e-4cc4-9fc6-13dfb87865ff
md"""
Some resources on time-series filters.

- [Time As a Variable: Time-Series Analysis](https://www.oreilly.com/library/view/data-analysis-with/9781449389802/ch04.html)
- [A Very Short Course on Time Series Analysis](https://bookdown.org/rdpeng/timeseriesbook/filtering-time-series.html)
"""

# ╔═╡ 53044c5e-b1b8-4433-8a0d-246ea257b991
function butterworth_filter(n, seq; cutoff_freq=0.2)
	zpk = digitalfilter(Lowpass(cutoff_freq), DSP.Butterworth(n))
	sos = convert(SecondOrderSections, zpk)
	DSP.filt(sos, seq)
end

# ╔═╡ 23791e95-85e1-4309-bfd8-0f7502fc3007
md"""
## Exponential Smoothing

"""

# ╔═╡ 0fe8f167-131b-444e-a39c-b5f258bea503
function exp_smoothing_iterate(exp_smo::Function, seq)
	[exp_smo(@view seq[1:i]) for i in 1:length(seq)]
end

# ╔═╡ a8b5e9af-5c63-4c95-81f8-06b4b9b88aa1
function exp_smoothing(α, X)
	s = X[1]
	for xi in 2:length(X)
		s = α * X[xi] + (1-α) * s
	end
	s
end

# ╔═╡ b298ff70-46c5-4e97-b412-efba0b9adada
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

# ╔═╡ 0cdebca8-23ab-40e9-98e1-a8f59d8fabf3
function triple_exp_smoothing(α, seq)
	s = seq[1]
	for xi in 2:length(seq)
		s = α * seq[xi] + (1-α) * s
	end
	s
end

# ╔═╡ be2f1b7f-b125-4b3b-b4a2-908bf80b0666
md"""
## Utility Funcs
- `unit_norm`/`unit_norm!`
- `series_sum`
"""

# ╔═╡ a7f44cdd-1ed9-42eb-b025-71f5a83a5cea
function unit_norm!(seq)
	seq .= (seq .- minimum(seq))./(maximum(seq) - minimum(seq))
end

# ╔═╡ 2f9e5cae-c3fa-4dbd-832d-7e394ff1f5eb
function unit_norm(seq)
	(seq .- minimum(seq))./(maximum(seq) - minimum(seq))
end

# ╔═╡ cbe6f964-bb11-4cf5-a29a-dcc7c085da99
butterworth_filter(1, unit_norm(data_mso))

# ╔═╡ be6f5969-2edf-4a11-9e8b-ffca31fcdd0f
function series_sum(γ, i)
	s = 0
	for i in 1:i
		s += γ^(i-1)*i
	end
	s
end

# ╔═╡ 0ff43cc5-9e9c-4f32-bfac-21ee642a4ca2
md"""
# Setup
"""

# ╔═╡ af44dffb-b488-4e10-8a93-02b3c531b806
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

# ╔═╡ c950f320-f7aa-45ec-a1ce-a17d779d3e12
CritterbotUtils = ingredients("utils/CritterbotUtils.jl").CritterbotUtils

# ╔═╡ a271aa6a-3bc5-4cc9-8ec3-d5917c343b71
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

# ╔═╡ 2bf7b355-ff82-4ba1-8645-b0e9b3a0c14c
# Setup Plots
let
	plotly()
	plot(rand(10), size=(5,5))
end

# ╔═╡ cb1b6734-5105-4764-b1ff-b7dfe4a6cfdb
PlutoUI.TableOfContents(title="Comp-GVFS", 
						indent=true, 
						aside=true, 
						depth=2)

# ╔═╡ 4071d625-6173-4cf6-8560-ec71aaac7335
import DataFrames

# ╔═╡ 749a58ba-bcc0-4739-9c96-d37c7dfe6312
critterbot_data = let
	col_names = CritterbotUtils.sensor_names()[CritterbotUtils.relevant_sensors_idx()]
	cb_data = CritterbotUtils.relevant_sensors()
	DataFrames.DataFrame(;(Symbol(n)=>d for (n, d) in zip(col_names, eachslice(cb_data, dims=2)))...)
end

# ╔═╡ a37c048e-9bbf-4722-ab45-4f39375fdbad
import FourierAnalysis

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
DSP = "~0.7.5"
DataFrames = "~1.3.3"
FourierAnalysis = "~1.2.0"
HDF5 = "~0.16.7"
MinimalRLCore = "~0.2.1"
Plots = "~1.23.6"
PlutoUI = "~0.7.38"
ProgressLogging = "~0.1.4"
Revise = "~3.3.3"
StatsPlots = "~0.14.30"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
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
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "6d4fa04343a7fc9f9cb9cff9558929f3d2752717"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.0.9"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "12fc73e5e0af68ad3137b886e3f7c1eacfca2640"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.17.1"

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

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "b153278a25dd42c65abbf4e62344f9d22e59191b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.43.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

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
git-tree-sha1 = "3e03979d16275ed5d9078d50327332c546e24e68"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.7.5"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "6c19003824cbebd804a51211fd3bbd81bf1ecad5"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.3"

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
git-tree-sha1 = "5a4168170ede913a2cd679e53c2123cb4b889795"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.53"

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
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

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

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

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
git-tree-sha1 = "cd6efcf9dc746b06709df14e462f0a3fe0786b1e"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.2+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

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
deps = ["Compat", "HDF5_jll", "Libdl", "Mmap", "Random", "Requires"]
git-tree-sha1 = "36df177c1ce5f399a8de959e5f4b75216fe6c834"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.16.7"

[[deps.HDF5_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "OpenSSL_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "bab67c0d1c4662d2c4be8c6007751b0b6111de5c"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.12.1+0"

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
git-tree-sha1 = "91b5dcf362c5add98049e6c29ee756910b03051d"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.3"

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
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "52617c41d2761cc05ed81fe779804d3b7f14fff7"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.13"

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
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "6f14549f7760d84b2db7a9b10b88cd3cc3025730"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.14"

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

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

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

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
git-tree-sha1 = "a970d55c2ad8084ca317a4658ba6ce99b7523571"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.12"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "dedbebe234e06e1ddad435f5c6f4b85cd8ce55f7"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.2.2"

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

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "8d958ff1854b166003238fe191ec34b9d592860a"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.8.0"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "ba8c0f8732a24facba709388c74ba99dcbfdda1e"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.0.0"

[[deps.NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "ded92de95031d4a8c61dfb6ba9adb6f1d8016ddd"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.10"

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
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

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
git-tree-sha1 = "e8185b83b9fc56eb6456200e873ce598ebc7f262"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.7"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "3b429f37de37f1fc603cc1de4a799dc7fbe4c0b6"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.0"

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
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun"]
git-tree-sha1 = "0d185e8c33401084cab546a756b387b15f76720c"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.23.6"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "670e559e5c8e191ded66fa9ea89c97f10376bb4c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.38"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "MutableArithmetics", "RecipesBase"]
git-tree-sha1 = "0107e2f7f90cc7f756fee8a304987c574bbd7583"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "3.0.0"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "28ef6c7ce353f0b35d0df0d5930e0d072c1f5b9b"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.1"

[[deps.PosDefManifold]]
deps = ["LinearAlgebra", "Statistics"]
git-tree-sha1 = "2010295335c00d7465571c6ee5e1eac5dc4a25e3"
uuid = "f45a3650-5c51-11e9-1e9a-133aa5e309cf"
version = "0.4.9"

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
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

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

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "4d4239e93531ac3e7ca7e339f15978d0b5149d03"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.3.3"

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

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "6a2f7d70512d205ca8c7ee31bfa9f142fe74310c"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.12"

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
git-tree-sha1 = "5ba658aeecaaf96923dce0da9e703bd1fe7666f9"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "cd56bf18ed715e8b09f06ef8c6b781e6cdc49911"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c82aaa13b44ea00134f8c9c89819477bd3986ecd"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.3.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5950925ff997ed6fb3e985dcce8eb1ba42a0bbe7"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.18"

[[deps.StatsPlots]]
deps = ["Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "e1e5ed9669d5521d4bbdd4fab9f0945a0ffceba2"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.30"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "57617b34fa34f91d536eb265df67c2d4519b8b98"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.5"

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
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

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
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

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
# ╠═ca3a10fa-511f-46ea-b71a-40aae51e7f34
# ╠═b927266d-01b0-4fb0-b8e2-03a351b4bb9f
# ╠═93585f0c-256a-43a6-9c54-8d39791b9b44
# ╠═f31e525c-85be-4c9f-8388-5f50790d9764
# ╠═cebeb2d9-e1b6-467e-ac54-32f405f5715e
# ╠═6cb07467-a793-4cf3-9afd-361eb487b579
# ╠═a896d5ce-84f1-41c3-aa7f-45ea40da3689
# ╠═b9f4121a-32a1-4d6d-b831-8e7ffedf8846
# ╠═82757b09-e34d-4d38-94e4-f1bf34a7a242
# ╠═eefb17c9-3e54-4cf4-9b98-73d7b73553d8
# ╠═291f437e-ab7a-45d3-983d-89881b4a7267
# ╠═d0a4b432-550a-4536-87a0-42abbf66ada8
# ╠═45ff69fc-9b2a-456f-b894-8522748c55a2
# ╠═054aecaa-9371-42a0-96c6-1dc9d03f8950
# ╟─bc4b2237-f34d-4909-a5c1-0b296b0921f5
# ╠═2b2e79d0-01aa-4a55-9bc3-4f81dacae075
# ╠═a8db4a35-0fe5-4fee-b9d5-81b6a2dd43d2
# ╠═ec59e1d0-4c36-42c2-a774-36723b425be0
# ╠═95e3fab5-c896-414a-bb1f-88f59d86344a
# ╠═cde0a299-2d15-4e67-90e5-51000f687852
# ╠═b2f12306-6e74-41f4-b401-b4fdee723b09
# ╠═29c34186-3232-4fc2-939e-4cb62cd62e12
# ╠═be716093-89f3-4933-b992-c166e70239d5
# ╠═c950f320-f7aa-45ec-a1ce-a17d779d3e12
# ╠═749a58ba-bcc0-4739-9c96-d37c7dfe6312
# ╟─3848090d-3b2f-4e76-8ae6-05da81517e33
# ╟─8c08b2d0-7d4d-4900-af08-c1572f01a149
# ╠═6fce8cae-55b6-4a36-b2bf-03a25634b38e
# ╠═c5807b71-26fd-46d3-aedc-65809d687e38
# ╠═843c7fa8-8894-43e7-bd0b-20498727cc4f
# ╠═9077cc84-3975-4494-95c1-90f0e11c9f2e
# ╟─f7426601-f710-4d6d-a100-0f1e5c0fa1a4
# ╟─ed399039-a185-4f8e-9ee0-eca1df7c1dfc
# ╠═d53e0ee5-baef-4891-a720-c54681af163c
# ╠═3e493422-8c49-4075-9d91-2826f194a1bd
# ╠═19733780-8f1c-4934-b38b-fa876069478e
# ╠═2b8eaa2a-16f4-4bde-9df6-4a05ba2a445d
# ╠═6d20cf86-c4c7-4977-9db2-07bece906045
# ╠═7a913e09-6ce3-4452-a63e-551f768495ec
# ╠═e1d9aa7b-f371-4ea9-8e5a-30d69fd20fce
# ╠═c515c139-a8ec-4447-901d-0f525d0017d9
# ╠═68ac9061-cdbd-41b1-9f4b-b352874952bb
# ╠═17130de6-9f7d-4d0a-9e43-e8dbad2726af
# ╠═d7fb81bd-970d-4091-85fc-48a364ca57ef
# ╟─670db08e-c9af-4c18-a86f-d37ceef09f9c
# ╠═b7a7b85e-d489-4a3a-b52f-5fec58737b0f
# ╠═53cd97e4-5ce8-4096-b8ba-b61880f80b57
# ╟─d5326af3-e9e3-4b75-8329-4f074f9fb37d
# ╠═0064e10a-2c86-4840-ac27-044341321021
# ╠═840ab122-fae5-4870-993e-ab7119fa4cfe
# ╠═5b18e194-720c-4cb7-b47b-88147795bddb
# ╠═97fd8519-3033-49a5-bf19-81f982a5d6a3
# ╠═a8d0b5c3-6a81-4380-a06c-66a2baae8394
# ╠═9bb09cae-6051-44c3-996a-0ccb14902db2
# ╠═461e5bc1-bc76-4f79-8a28-5f95265d9e3d
# ╟─37b2ef37-2136-4363-8df2-967f7df12475
# ╟─c961895d-ba02-444a-85a6-9b2128c6c572
# ╠═08c0950c-bd12-4ae8-b62d-057c67a5c83c
# ╠═66892139-2087-401d-83b3-a7205813b1f7
# ╠═7a6a45ee-bcfc-4833-8955-3022d6d77210
# ╠═4af3253d-5fba-4d4f-bd08-66d6c7e55347
# ╠═9a1d6978-e078-44c3-aac4-7039f368d16b
# ╠═d24de518-5d73-410f-89df-68668a2b2d99
# ╠═af68d866-f508-4ea4-9597-f3378443a932
# ╠═3818803c-b121-4128-894b-44749d8fc214
# ╠═d57e9370-3ffb-4f3e-94e0-212968c2113d
# ╠═6bee6c47-d0e9-4b39-9807-f1f90d8a98af
# ╠═193fdee2-008e-4cc4-9fc6-13dfb87865ff
# ╠═330cb40c-0ea1-41c5-bed3-c3ade059555b
# ╠═53044c5e-b1b8-4433-8a0d-246ea257b991
# ╠═cbe6f964-bb11-4cf5-a29a-dcc7c085da99
# ╠═23791e95-85e1-4309-bfd8-0f7502fc3007
# ╠═0fe8f167-131b-444e-a39c-b5f258bea503
# ╠═a8b5e9af-5c63-4c95-81f8-06b4b9b88aa1
# ╠═b298ff70-46c5-4e97-b412-efba0b9adada
# ╠═0cdebca8-23ab-40e9-98e1-a8f59d8fabf3
# ╠═be2f1b7f-b125-4b3b-b4a2-908bf80b0666
# ╠═a7f44cdd-1ed9-42eb-b025-71f5a83a5cea
# ╠═2f9e5cae-c3fa-4dbd-832d-7e394ff1f5eb
# ╠═be6f5969-2edf-4a11-9e8b-ffca31fcdd0f
# ╟─0ff43cc5-9e9c-4f32-bfac-21ee642a4ca2
# ╠═af44dffb-b488-4e10-8a93-02b3c531b806
# ╠═a271aa6a-3bc5-4cc9-8ec3-d5917c343b71
# ╠═2bf7b355-ff82-4ba1-8645-b0e9b3a0c14c
# ╠═cb1b6734-5105-4764-b1ff-b7dfe4a6cfdb
# ╠═4c5ea1f7-ddd5-44fe-be88-b582803dbe75
# ╠═fc6e55f1-60a9-4c82-892e-9555f2f70ba1
# ╠═26c6d74e-8c06-4d60-b52f-cbe8b05ecbee
# ╠═bf2b7016-167b-4c8d-8616-a5b8847223f3
# ╠═4071d625-6173-4cf6-8560-ec71aaac7335
# ╠═a37c048e-9bbf-4722-ab45-4f39375fdbad
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
