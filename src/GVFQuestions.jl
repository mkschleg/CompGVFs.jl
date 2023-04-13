


import MinimalRLCore
import Random

# These Need to be declared first!


struct GVFQuestion{C, Π, Γ}
    c::C
    π::Π
    γ::Γ
end


"""
    AbstractQPolicy

This policy has both a get_value and get_action policy. It is supposed to be used with
state-action value functions. Will be used in a BDemon.

Interface:
get_value(π, q, a_t)
get_action(rng, π, q)
"""
abstract type AbstractQPolicy end

const BDemon{C, Π, Γ} = GVFQuestion{C, Π, Γ} where {C, Π<:AbstractQPolicy, Γ}


# What makes a BDemon but a GVF Question with a AbstractActingPolicy
# mutable struct BDemon{W, C, Π, Γ}
#     c::C
#     π::Π
#     γ::Γ
#     BDemon(num_feats::Int, num_actions::Int, cumulant, policy, discount) = 
# 	new{Matrix{Float64}, typeof(cumulant), typeof(policy), typeof(discount)}(
#             zeros(num_actions, num_feats), zeros(num_actions, num_feats), cumulant, policy, discount)
# end

# function MinimalRLCore.start!(bdemon::BDemon, s_t, x_t=s_t)
#     bdemon.z .= 0
# end
# function MinimalRLCore.is_terminal(bdemon::BDemon, s_t, x_t)
#     get_value(bdemon.γ, s_t, x_t) == 0
# end

predict(bdemon::BDemon{<:AbstractMatrix}, x::Int) = begin; 
    bdemon.w[:, x]
end

# ╔═╡ 33db32af-2e34-4dad-abcc-e023349aae70
predict(bdemon::BDemon{<:AbstractMatrix}, a::Int, x::Int) = begin; 
    bdemon.w[a, x]
end

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
get_value(rsc::RescaleCumulant, args...) = 
    get_value(rsc.c, args...)*(1-rsc.γ)

struct ThresholdCumulant{C, F}
    c::C
    θ::F
end
get_value(rsc::ThresholdCumulant, args...) = 
    get_value(rsc.c, args...) >= rsc.θ ? 1 : 0

struct GVFCumulant{G}
    gvf::G
end
get_value(gvfc::GVFCumulant, o, x, p, r) = 
    predict(gvfc.gvf, x)

"""
    Policies and the Importance sampling ratio...
"""

function get_importance_ratio(target_policy, behavior_prob, s, a)
    get_value(target_policy, s, a) / behavior_prob
end

"""
    OnPolicy

This one is special.
"""
struct OnPolicy end # OnPolicy is kinda special. 
get_value(op::OnPolicy, args...) = 1

function get_importance_ratio(target_policy::OnPolicy, behavior_prob, s, a) # Special case for onpolicy
    # behavior_prob
    1
end


struct PersistentPolicy{A}
    action::A
end

get_value(pp::PersistentPolicy, s, a) = a == pp.action ? 1 : 0

struct RandomPolicy{AS}
    actions::AS
end

get_value(rp::RandomPolicy, args...) = 1//length(actions)

# Using BDemon as a policy
function get_value(bdemon::BDemon, s_t, a_t)
    get_value(bdemon.π, predict(bdemon, s_t), a_t)
end



struct ϵGreedy <: AbstractQPolicy # We kind of need this kind of policy?
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

struct GVFThreshTerminatingDiscount{F, G}
    γ::F
    gvf::G
    θ::Float64
end
get_value(fc::GVFThreshTerminatingDiscount, o, x::Int) = begin
    predict(fc.gvf, x) > fc.θ ? zero(typeof(fc.γ)) : fc.γ
end

mutable struct GVFThreshTerminatingMaxDiscount{F, G}
    γ::F
    gvf::G
    θ::Float64
    GVFThreshTerminatingMaxDiscount(γ, gvf) = new{typeof(γ), typeof(gvf)}(γ, gvf, -Inf)
end

get_value(fc::GVFThreshTerminatingMaxDiscount, o, x::Int) = begin
    pred = predict(fc.gvf, x)
    if pred > fc.θ || pred ≈ fc.θ
	fc.θ = pred
	return zero(typeof(fc.γ))
    else
	return fc.γ
    end
end
