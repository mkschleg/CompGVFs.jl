
import MinimalRLCore
import RecipesBase: RecipesBase, @recipe, @series
import Colors: Colors, @colorant_str
import Statistics: median

mutable struct GridWithWalls <: MinimalRLCore.AbstractEnvironment
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
    GridWithWalls(BASE_WALLS)
end

function NoWallsGridWorld(width, height)
    GridWithWalls(fill(0, height, width))
end

function NoWallsGridWorld()
    BASE_WALLS = [0 0 0 0 0 0 0 0 0 0 0;
                  0 0 0 0 0 0 0 0 0 0 0;
                  0 0 0 0 0 0 0 0 0 0 0;
                  0 0 0 0 0 0 0 0 0 0 0;
                  0 0 0 0 0 0 0 0 0 0 0;
               	  0 0 0 0 0 0 0 0 0 0 0;
                  0 0 0 0 0 0 0 0 0 0 0;
                  0 0 0 0 0 0 0 0 0 0 0;
                  0 0 0 0 0 0 0 0 0 0 0;
                  0 0 0 0 0 0 0 0 0 0 0;
                  0 0 0 0 0 0 0 0 0 0 0;]
    GridWithWalls(BASE_WALLS)
end

MinimalRLCore.is_terminal(env::GridWithWalls) = false

MinimalRLCore.get_reward(env::GridWithWalls) = 0

is_wall(env::GridWithWalls, state) = env.walls[state[1], state[2]]

Base.size(env::GridWithWalls, args...) = size(env.walls, args...)

MinimalRLCore.get_state(env::GridWithWalls, state = env.state) = begin
    (state[1] - 1) * size(env, 1) + state[2]
end

random_state(env::GridWithWalls, rng) = [rand(rng, 1:size(env.walls)[1]), rand(rng, 1:size(env.walls)[2])]

num_actions(env::GridWithWalls) = 4

get_states(env::GridWithWalls) = findall(x->x==false, env.walls)

MinimalRLCore.get_actions(env::GridWithWalls) = FourRoomsParams.ACTIONS

function MinimalRLCore.reset!(env::GridWithWalls, rng=nothing; kwargs...)
    state = random_state(env, rng)
    while env.walls[state[1], state[2]]
        state = random_state(env, rng)
    end
    env.state = state
    return state
end

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

# Used for dynamic programming if integrated.
# function _step(env::GridWithWalls, state, action, rng, kwargs...)
#     frp = env
#     next_state = copy(state)

#     if action == frp.UP
#         next_state[1] -= 1
#     elseif action == frp.DOWN
#         next_state[1] += 1
#     elseif action == frp.RIGHT
#         next_state[2] += 1
#     elseif action == frp.LEFT
#         next_state[2] -= 1
#     end

#     next_state[1] = clamp(next_state[1], 1, size(env.walls, 1))
#     next_state[2] = clamp(next_state[2], 1, size(env.walls, 2))
#     if is_wall(env, next_state)
#         next_state = state
#     end

#     return next_state, 0, false
# end

# # ╔═╡ 983c6296-db06-419c-81ed-eac04fa4db75
# function _step(env::GridWithWalls, state::CartesianIndex{2}, action)
#     array_state = [state[1], state[2]]
#     new_state, r, t = _step(env, array_state, action)
#     return CartesianIndex{2}(new_state[1], new_state[2]), r, t
# end

function Base.show(io::IO, env::GridWithWalls)
    model = fill("□", size(env.walls)...)
    model[env.walls] .= "▤"
    model[env.state[1], env.state[2]] = "◍"
    for row in eachrow(model)
	println(io, join(row, " "))
    end
end

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

@recipe function f(env::GridWithWalls, bdemon::ControlLearner)
    ticks := nothing
    foreground_color_border := nothing
    grid := false
    legend := false
    aspect_ratio := 1
    xaxis := false
    yaxis := false
    yflip := false

    colored_arrows --> true

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

    colors = []
    
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
			push!(colors, color_scheme[1])
		    elseif a == env.DOWN
			push!(q_x, med_sqr_j)
			push!(q_y, med_sqr_i - 1)
			push!(q_u, 0)
			push!(q_v, -1*SIZE/2 + 6)
			push!(colors, color_scheme[2])
		    elseif a == env.RIGHT
			push!(q_x, med_sqr_j + 1)
			push!(q_y, med_sqr_i)
			push!(q_u, 1*SIZE/2 - 6)
			push!(q_v, 0)
			push!(colors, color_scheme[5])
		    elseif a == env.LEFT
			push!(q_x, med_sqr_j - 1)
			push!(q_y, med_sqr_i)
			push!(q_u,  -1*SIZE/2 + 6)
			push!(q_v, 0)
			push!(colors, color_scheme[4])
		    end
		end
	    end
	end
    end

    arrow_c = if plotattributes[:colored_arrows]
        repeat(colors, inner=4)
    else
        :black
    end
    
    @series begin
	seriestype := :quiver
	arrow := true
	c := arrow_c
        seriescolor := arrow_c
        linecolor := arrow_c
        markercolor := arrow_c
	linewidth := 2
        legend := false
	label := nothing
        
	gradient := (q_u, q_v)
	q_x, q_y
    
    end
    
end
