
import MinimalRLCore
import Random

struct GVF{C, Π, Γ}
    c::C
    π::Π
    γ::Γ
end

# GVF(cumulant, policy, discount) = 
#     GVF(cumulant, policy, discount)


# predict(gvf::GVF{<:AbstractVector}, x::AbstractVector{<:Number}) = 
#     dot(gvf.w, x)

# predict(gvf::GVF{<:AbstractVector}, x::AbstractVector{Int}) = begin; 
#     w = gvf.w; 
#     ret = sum(view(w, x))
# end

# predict(gvf::GVF{<:AbstractVector}, x::Int) = begin; 
#     gvf.w[x]
# end

# predict(gvf::GVF{<:AbstractMatrix}, x::AbstractVector{Int}) = begin; 
#     gvf.w[x]
# end

const Horde = Vector{<:GVF}
# predict(horde::Horde, x) = [predict(gvf, x) for gvf in horde]


mutable struct BDemon{C, Π, Γ}
    c::C
    π::Π
    γ::Γ
end


function MinimalRLCore.is_terminal(bdemon::BDemon, s_t, x_t)
    get_value(bdemon.γ, s_t, x_t) == 0
end

# predict(bdemon::BDemon{<:AbstractMatrix}, x::Int) = begin; 
#     bdemon.w[:, x]
# end

# # ╔═╡ 33db32af-2e34-4dad-abcc-e023349aae70
# predict(bdemon::BDemon{<:AbstractMatrix}, a::Int, x::Int) = begin; 
#     bdemon.w[a, x]
# end

function get_action(demon::BDemon, x_t)
    get_action(demon.π, predict(demon, x_t))
end

function get_action(rng::Random.AbstractRNG, demon::BDemon, x_t)
    get_action(rng, demon.π, predict(demon, x_t))
end

# ╔═╡ 5ab59373-6a80-46b2-8687-ade15aa31b5e
"""
    Parameter Functions
"""

# ╔═╡ c9f0f1e2-d596-45d4-9a6b-ead881e615f3
"""
    Cumulants
"""

struct FeatureCumulant
    idx::Int
end

get_value(fc::FeatureCumulant, o, x, p, r) = x[fc.idx]
get_value(fc::FeatureCumulant, o, x::Int, p, r) = fc.idx == x
get_value(fc::FeatureCumulant, o, x::Vector{Int}, p, r) = fc.idx ∈ x

struct ObservationCumulant
    idx::Int
end
get_value(oc::ObservationCumulant, o, x, p, r) = o[oc.idx]

struct PredictionCumulant
    idx::Int
end
get_value(pc::PredictionCumulant, o, x, p, r) = p[pc.idx]


struct RescaleCumulant{C, F}
    c::C
    γ::F
end
get_value(rsc::RescaleCumulant, o, x, p, r) = 
    get_value(rsc.c, o, x, p, r)*(1-rsc.γ)

struct ThresholdCumulant{C, F}
    c::C
    θ::F
end
get_value(rsc::ThresholdCumulant, o, x, p, r) = 
    get_value(rsc.c, o, x, p, r) >= rsc.θ ? 1 : 0



# struct GVFCumulant{G<:GVF}
#     gvf::G
# end
# get_value(gvfc::GVFCumulant, o, x, p, r) = 
#     predict(gvfc.gvf, x)

"""
    Policies
"""

struct OnPolicy end
get_value(op::OnPolicy, args...) = 1

struct PersistentPolicy{A}
    action::A
end

get_value(pp::PersistentPolicy, s, a) = a == pp.action ? 1 : 0

struct RandomPolicy{AS}
    actions::AS
end

get_value(rp::RandomPolicy, args...) = 1//length(actions)

# Using BDemon as a policy
# function get_value(bdemon::BDemon, s_t, a_t)
#     get_value(bdemon.π, predict(bdemon, s_t), a_t)
# end

struct ϵGreedy
    ϵ::Float32
end

function get_value(π::ϵGreedy, q, a_t)
    idx = findall(==(maximum(q)), q)
    if a_t ∈ idx
	(1 - π.ϵ) / length(idx)
    else
	π.ϵ / length(q)
    end
end

function get_action(rng::Random.AbstractRNG, π::ϵGreedy, q)
    if rand() < π.ϵ
	rand(rng, 1:length(q))
    else
	rand(rng, findall(==(maximum(q)), q))
    end
end
get_action(π, q) = get_action(Random.default_rng(), π, q)


"""
    Discounts
"""

struct ConstantDiscount{F}
    γ::F
end
get_value(cd::ConstantDiscount, args...) = cd.γ

struct TerminatingDiscount{F}
    γ::F
    idx::Int
end
get_value(fc::TerminatingDiscount, o, x::Vector{Int}) = fc.idx ∈ x ? zero(typeof(fc.γ)) : fc.γ
get_value(fc::TerminatingDiscount, o, x::Int) = fc.idx == x ? zero(typeof(fc.γ)) : fc.γ




