

struct IndexVHordeLearner <: AbstractLearner
    l::VHordeLearner{<:LinearCollection}
    idx::Int
end

predict(l::IndexVHordeLearner, args...) =
    predict(l.l.answer.funcs[l.idx], args...)
    

struct GVFCumulant{G<:IndexVHordeLearner}
    gvf::G
end
get_value(gvfc::GVFCumulant, o, x, p, r) = 
    predict(gvfc.gvf, x)



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
