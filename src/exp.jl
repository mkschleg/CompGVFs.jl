# Simple experiment functions
import ProgressLogging: @progress
import MinimalRLCore: MinimalRLCore, start!, step!, is_terminal

function fourrooms_experiment!(learner::HordeLearner,
                               num_steps;
                               seed=1, kwargs...)
    rng = Random.Xoshiro(seed)

    env = FourRooms() #CycleWorld(env_size, partially_observable=false)
    
    s_t = start!(env, rng)
    
    @progress for step in 1:num_steps

	a_t = rand(rng, env.ACTIONS)
        μ_t = 1/length(env.ACTIONS)

	s_tp1, r_tp1, _ = step!(env, a_t)
        
        update!(learner, s_t, a_t, s_tp1, r_tp1)
	# p_tp1 = predict(learner, s_tp1)
	# update!(lu, horde, s_t, s_t, a_t, 
	# 	1/length(env.ACTIONS), s_tp1, s_tp1, 
	# 	r_tp1, p_tp1)

	s_t = copy(s_tp1)
    end
    env_size = size(FourRooms())
    env_feat_size = env_size[1] * env_size[2] 
    p = [predict(learner, x) for x in 1:env_feat_size]
    
end

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


function fourrooms_behavior!(
    learner, 
    num_steps; seed = 1, kwargs...)

    rng = Random.Xoshiro(seed)
    
    env = FourRooms() #CycleWorld(env_size, partially_observable=false)

    total_steps = 0

    while total_steps < num_steps
	s_t = start!(env, rng)
	start!(learner, s_t)
	while is_terminal(learner, s_t) == false
	    a_t = get_action(rng, learner, s_t)

	    s_tp1, r_tp1, _ = MinimalRLCore.step!(env, a_t)
            update!(learner, s_t, a_t, s_tp1, r_tp1)

	    total_steps += 1
	    s_t = copy(s_tp1)
	    total_steps < num_steps || break
	end
	# bdemon.z .= 0
    end
    env_size = size(FourRooms())
    env_feat_size = env_size[1] * env_size[2] 
    p = [predict(learner, x) for x in 1:env_feat_size]
    
    p
end


function fourrooms_behavior_offpolicy!(
    learner,
    num_steps; kwargs...)

    env = FourRooms() #CycleWorld(env_size, partially_observable=false)

    total_steps = 0
    while total_steps < num_steps 
	s_t = start!(env, Random.default_rng())

        while true
	    a_t = rand(1:4)

	    s_tp1, r_tp1, _ = step!(env, a_t)
            update!(learner, s_t, a_t, s_tp1, r_tp1)

	    total_steps += 1
            s_t = copy(s_tp1)
	
	    (is_terminal(learner, s_tp1) == false) || break
	    total_steps < num_steps || break
	end
    end
    @info total_steps
    env_size = size(FourRooms())
    env_feat_size = env_size[1] * env_size[2] 
    p = [predict(learner, x) for x in 1:env_feat_size]
    
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
        while true
	    a_t = rand(1:4)

	    s_tp1, r_tp1, _ = step!(env, a_t)
	    p_tp1 = predict(bdemon, s_tp1)
	    update!(lu, bdemon, s_t, s_t, a_t, nothing, s_tp1, s_tp1, r_tp1, p_tp1)

	    total_steps += 1
	    s_t = copy(s_tp1)
	
	    (is_terminal(bdemon, s_tp1, s_tp1) == false) || break
	    total_steps < num_steps || break
	end
    end
    @info total_steps
    env_size = size(FourRooms())
    env_feat_size = env_size[1] * env_size[2] 
    p = [predict(bdemon, x) for x in 1:env_feat_size]
    
    p
end
