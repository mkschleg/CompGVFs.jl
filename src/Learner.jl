
struct Learner{FT, H, A, LU, FC, S}
    horde::H
    answer::A
    update::LU
    feat_constructor::FC
    state::S
    Learner{FT}(horde, approximator, update, identity, state) where FT =
        new{FT, typeof(horde), typeof(approximator), typeof(update), typeof(identity), typeof(state)}(
            horde, approximator, update, identity, state
        )
end
Learner{FT}(gvf, approximator, update) where {FT} = Learner{FT}(gvf, approximator, update, identity, setup(update, gvf, approximator))

struct QFunction end
struct VFunction end

const ControlLearner{A, LU, FC, S} = Learner{QFunction, <:BDemon, LU, FC, S}
const HordeLearner{FT, A, LU, FC, S} = Learner{FT, <:Horde, LU, FC, S} #When do we do state-actions with a horde of GVFs?

function QLinearLearner(in, num_actions, horde, update)
    if horde isa AbstractVector
        Learner{QFunction}(horde, Linear(in, length(horde)*num_actions), update)
    else
        Learner{QFunction}(horde, Linear(in, num_actions), update)
    end
end

function VLinearLearner(in, horde, update)
    if horde isa AbstractVector
        Learner{VFunction}(horde, Linear(in, length(horde)), update)
    else
        Learner{VFunction}(horde, Linear(in), update)
    end
end

function MinimalRLCore.start!(learner::Learner, s_t)
    # start!(learner.horde, s_t)
end

function get_action(learner::ControlLearner, x_t)
    get_action(Random.default_rng(), learner, x_t)
end

function get_action(rng::Random.AbstractRNG, learner::ControlLearner, x_t)
    q = predict(learner, x_t)
    get_action(rng, learner.horde.π, q)
end

function update!(learner::ControlLearner, o_t, a_t, o_tp1, r_tp1)
    bdemon = learner.horde
    x_t = learner.feat_constructor(o_t)
    x_tp1 = learner.feat_constructor(o_tp1)
    p_tp1 = predict(learner.answer, x_tp1)

    gvf = learner.horde 

    γ_t, γ_tp1 = if gvf.γ isa AbstractFloat
        eltype(w)(gvf.γ)
    else
        get_value(gvf.γ, o_t, x_t), get_value(gvf.γ, o_tp1, x_tp1)
    end

    c = get_value(gvf.c, o_tp1, x_tp1, p_tp1, r_tp1)
    
    update!(learner.update, bdemon,
            learner.answer, learner.state,
            x_t, x_tp1, a_t, c, γ_t, γ_tp1)
    #o_t, x_t, a_t, nothing, o_tp1, x_tp1, r_tp1, p_tp1)
end

function get_is(horde::Horde, o_t, a_t)
end

function get_discount()
end

function get_cumulant()
end

function update!(learner::HordeLearner, o_t, a_t, μ_t, o_tp1, r_tp1)
    bdemon = learner.horde
    x_t = learner.feat_constructor(o_t)
    x_tp1 = learner.feat_constructor(o_tp1)
    p_tp1 = predict(learner.answer, x_tp1)

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
    
    update!(learner.update, bdemon,
            learner.answer, learner.state,
            x_t, x_tp1, a_t, c, γ_t, γ_tp1)
    #o_t, x_t, a_t, nothing, o_tp1, x_tp1, r_tp1, p_tp1)
end

function predict(learner::Learner, args...)
    predict(learner.answer, args...)
end

function MinimalRLCore.is_terminal(learner::ControlLearner, obs)
    x = learner.feat_constructor(obs)
    is_terminal(learner.horde, obs, x)
end
