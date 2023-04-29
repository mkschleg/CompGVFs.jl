### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# ╔═╡ 903d11ae-186b-4dcc-bab5-14d9b1455069
begin
	import Pkg
	using Revise
	Pkg.activate(Pkg.Base.current_project())
	Pkg.instantiate()
end

# ╔═╡ dacb15d9-7bbd-4195-b8b6-9f3053f1052b
using MinimalRLCore

# ╔═╡ caaec4e0-67ba-4a38-9815-98faa726998d
import CompGVFs

# ╔═╡ 5a44cc9f-4e4c-492e-8aee-870fc4e52e6e
function fourrooms_heatmap_valuefunction(p::AbstractVector{<:AbstractFloat})
	heatmap(reshape(p, 11, 11)[:, end:-1:1]')
end

# ╔═╡ c95ff355-9c70-410f-b852-f7c3004a773e
function fourrooms_heatmap_policy(p::AbstractVector{<:AbstractVector})
	plt_up = fourrooms_heatmap_valuefunction(getindex.(p, 1))
	title!(plt_up, "Up")
	plt_right = fourrooms_heatmap_valuefunction(getindex.(p, 2))
	title!(plt_right, "Right")
	plt_down = fourrooms_heatmap_valuefunction(getindex.(p, 3))
	title!(plt_down, "Down")
	plt_left = fourrooms_heatmap_valuefunction(getindex.(p, 4))
	title!(plt_left, "Left")

	plot(plt_up, plt_right, plt_down, plt_left)
	# heatmap(reshape(p, 11, 11)[:, end:-1:1]')
end

# ╔═╡ d107cd3a-b441-4a42-8c62-90034bcb028c
let
	env = CompGVFs.FourRooms()
	env.state = [11, 1]
	MinimalRLCore.get_state(env)
end

# ╔═╡ ce18fa61-a224-4230-a0cb-55d84b9d1295
fr_bdeomon_11_pred, fr_bdemon_11 = let
	env_size = size(CompGVFs.FourRooms())
	env_feat_size = env_size[1] * env_size[2] 
	bdemon = CompGVFs.GVFQuestion(
		CompGVFs.FeatureCumulant(11), 
		CompGVFs.ϵGreedy(0.1),
		CompGVFs.TerminatingDiscount(0.9, 11))
	env_feat_size, 4
	
	CompGVFs.fourrooms_behavior!(bdemon, 1_000_000, CompGVFs.Qλ(0.1, 0.9)), bdemon
end

# ╔═╡ 6fb15522-98e6-46a4-832c-432da771e8a9
fourrooms_heatmap_policy(fr_bdeomon_11_pred)

# ╔═╡ 3dde5483-4e7f-44a2-a21e-801bfbe4b623
plot(CompGVFs.FourRooms(), fr_bdemon_11)

# ╔═╡ 84316175-c9ac-491e-ac2f-072e7fc5f93c
let
	if false
		plt = plot(CompGVFs.FourRooms(), fr_bdemon_11)
		savefig(plt, "../fourrooms_example/bdemon_11.png")
		plt = fourrooms_heatmap_policy(fr_bdeomon_11_pred)
		savefig(plt, "../fourrooms_example/bdemon_11_heatmap.png")
	end
end

# ╔═╡ 9aa92437-6df6-4a95-8810-e295423b20b6
fr_bd_hrd, fr_bd_p = let	
	num_steps = 2_000_000
	lu = CompGVFs.TDλ(0.01, 0.9)
	γ = 0.9
	env_size = size(CompGVFs.FourRooms())
	env_feat_size = env_size[1] * env_size[2] 
	horde, p = CompGVFs.fourrooms_experiment(num_steps, lu) do 
		[[CompGVFs.GVF(env_feat_size, 
			CompGVFs.FeatureCumulant(11),
			fr_bdemon_11, 
			# ConstantDiscount(γ))
			CompGVFs.TerminatingDiscount(γ, 11))
		];
		[CompGVFs.GVF(env_feat_size, 
			CompGVFs.RescaleCumulant(
				CompGVFs.PredictionCumulant(i), 
				γ),
			fr_bdemon_11, 
			# ConstantDiscount(γ))
			CompGVFs.TerminatingDiscount(γ, 11))
			for i in 1:11]]
	end
	horde, p
end

# ╔═╡ d89eec44-5df2-4a46-b9db-8e68adbe6c3e
fourrooms_heatmap_valuefunction(getindex.(fr_bd_p, 5))

# ╔═╡ 206a7861-dca5-433f-b5af-f2c3449f97bc
let
	if false
		for i in 1:length(fr_bd_p[1])
			plt = fourrooms_heatmap_valuefunction(getindex.(fr_bd_p, i))
			savefig(plt, "../fourrooms_example/comp_values_$(i).png")
		end
	end
end

# ╔═╡ 6bd6bbc9-b1f6-4a72-abbb-874bb1c64243
fr_comp_bdemon_max_p, fr_comp_bdemon_max = let	
	env_size = size(CompGVFs.FourRooms())
	env_feat_size = env_size[1] * env_size[2] 
	bdemon = CompGVFs.BDemon(
		env_feat_size, 
		4, 
		CompGVFs.GVFCumulant(fr_bd_hrd[5]),
		# FeatureCumulant(11), 
		CompGVFs.ϵGreedy(0.1),
		CompGVFs.GVFThreshTerminatingMaxDiscount(0.9, fr_bd_hrd[4]))
	
	CompGVFs.fourrooms_behavior_offpolicy!(bdemon, 500_000, CompGVFs.Qλ(0.01, 0.9)), bdemon
end

# ╔═╡ fb9fd7af-c918-4632-b695-6396359fe480
plot(CompGVFs.FourRooms(), fr_comp_bdemon_max)

# ╔═╡ c792c013-ea1a-435b-8cb6-f284c754417d
let
	if false
		plt = plot(CompGVFs.FourRooms(), fr_comp_bdemon_max)
		savefig(plt, "../fourrooms_example/comp_bdemon_max.pdf")
		plt = fourrooms_heatmap_policy(fr_comp_bdemon_max_p)
		savefig(plt, "../fourrooms_example/comp_bdemon_max_heatmap.pdf")
	end
end

# ╔═╡ 2fb9bfc2-1df2-4c96-b632-533cc41b463c
fourrooms_heatmap_policy(fr_comp_bdemon_max_p)

# ╔═╡ 6ad53250-6177-40f2-bdad-be3851294172
fr_comp_bdemon_const_p, fr_comp_bdemon_const = let	
	env_size = size(CompGVFs.FourRooms())
	env_feat_size = env_size[1] * env_size[2] 
	bdemon = CompGVFs.BDemon(
		env_feat_size, 
		4, 
		CompGVFs.GVFCumulant(fr_bd_hrd[5]),
		# FeatureCumulant(11), 
		CompGVFs.ϵGreedy(0.1),
		CompGVFs.ConstantDiscount(0.9))
	
	CompGVFs.fourrooms_behavior_offpolicy!(bdemon, 500_000, CompGVFs.Qλ(0.01, 0.9)), bdemon
end

# ╔═╡ cd3b5c29-6a2c-45b2-afc8-1aa59e96f519
plot(CompGVFs.FourRooms(), fr_comp_bdemon_const)

# ╔═╡ 0037bb06-d6b5-45ae-84b9-31af35683d25
fourrooms_heatmap_policy(fr_comp_bdemon_const_p)

# ╔═╡ 2bb526c7-576f-421a-bc1c-0580d09dd9e8
let
	if false
		plt = plot(CompGVFs.FourRooms(), fr_comp_bdemon_const)
		savefig(plt, "../fourrooms_example/comp_bdemon_const.png")
		plt = fourrooms_heatmap_policy(fr_comp_bdemon_const_p)
		savefig(plt, "../fourrooms_example/comp_bdemon_const_heatmap.png")
	end
end

# ╔═╡ dc5b5f44-5e4a-4eda-9c6d-c378c79e553d
begin
struct CombinedDisc{D1, D2}
	d1::D1
	d2::D2
end
# (d::CombinedDisc)(args...) = CompGVFs.get_value(d, args...)
end

# ╔═╡ 808e8e40-d60a-4f17-a605-5b6b287a1c18


# ╔═╡ 1092fdbc-8987-451f-a8d0-dcbe48a2f340
begin
struct CombinedCumulant{C1, C2}
	c1::C1
	c2::C2
end
# (c::CombinedCumulant)(args...) = CompGVFs.get_value(c, args...)
end

# ╔═╡ 7341e0f7-ab92-436e-ad77-2900122b60c3
struct NegateCumulant{C}
	c::C
end

# ╔═╡ 0db6b442-f0e9-4057-9b6b-0db16ac250eb
CompGVFs.get_value(c::NegateCumulant, o, x, p, r) = -1 * CompGVFs.get_value(c.c, o, x, p, r)

# ╔═╡ 87bfaf12-a965-48ff-b243-eae9902e6d6f
CompGVFs.get_value(c::CombinedCumulant, o, x, p, r) = CompGVFs.get_value(c.c1, o, x, p, r) + CompGVFs.get_value(c.c2, o, x, p, r)

# ╔═╡ e0f7130b-b26f-49cc-bea1-132afc4eee6d
CompGVFs.get_value(d::CombinedDisc, o, x) = (CompGVFs.get_value(d.d1, o, x) * CompGVFs.get_value(d.d2, o, x))/d.d1.γ

# ╔═╡ 967863c7-b4b7-48d7-a811-42e8c3f1f62f
fr_bdeomon_joint_pred, fr_bdemon_joint = let
	env_size = size(CompGVFs.FourRooms())
	env_feat_size = env_size[1] * env_size[2] 
	
	bdemon = CompGVFs.BDemon(env_feat_size, 4, 
		CombinedCumulant(
			CompGVFs.FeatureCumulant(11), 
			CompGVFs.FeatureCumulant(111)),
		CompGVFs.ϵGreedy(0.1),
		CombinedDisc(
			CompGVFs.TerminatingDiscount(0.9, 11), 
			CompGVFs.TerminatingDiscount(0.9, 111)))
	CompGVFs.fourrooms_behavior!(bdemon, 1_000_000, CompGVFs.Qλ(0.1, 0.9)), bdemon
end

# ╔═╡ 49c6eb84-7bf5-4cd1-aa64-b0a34abf4af4
fourrooms_heatmap_policy(fr_bdeomon_joint_pred)

# ╔═╡ abd2f153-9aeb-4aa6-a368-c561b6f27701
plot(CompGVFs.FourRooms(), fr_bdemon_joint)

# ╔═╡ 604bc156-047c-47f7-9c69-4d7dc0fc9692
let
	if false
		plt = plot(CompGVFs.FourRooms(), fr_bdemon_joint)
		savefig(plt, "../fourrooms_example/joint_bdemon.pdf")
		plt = fourrooms_heatmap_policy(fr_bdeomon_joint_pred)
		savefig(plt, "../fourrooms_example/joint_bdemon_heatmap.png")
	end
end

# ╔═╡ 8e93d983-143c-4d98-9702-e2ff956a425d
fr_bd_joint_hrd, fr_bd_joint_p = let	
	num_steps = 2_000_000
	lu = CompGVFs.TDλ(0.01, 0.9)
	γ = 0.9
	env_size = size(CompGVFs.FourRooms())
	env_feat_size = env_size[1] * env_size[2] 
	horde, p = CompGVFs.fourrooms_experiment(num_steps, lu) do 
		[[CompGVFs.GVF(env_feat_size, 
			CombinedCumulant(
				CompGVFs.FeatureCumulant(11), 
				CompGVFs.FeatureCumulant(111)),
			fr_bdemon_joint, 
			# ConstantDiscount(γ))
			CombinedDisc(
				CompGVFs.TerminatingDiscount(0.9, 11), 
				CompGVFs.TerminatingDiscount(0.9, 111)))
		];
		[CompGVFs.GVF(env_feat_size, 
			CompGVFs.RescaleCumulant(
				CompGVFs.PredictionCumulant(i), 
				γ),
			fr_bdemon_joint, 
			# ConstantDiscount(γ))
			CombinedDisc(
				CompGVFs.TerminatingDiscount(0.9, 11), 
				CompGVFs.TerminatingDiscount(0.9, 111)))
			for i in 1:11]]
	end
	horde, p
end

# ╔═╡ 4dd55820-2f55-4cd4-bc14-54ad6a0418b5
fourrooms_heatmap_valuefunction(getindex.(fr_bd_joint_p, 5))

# ╔═╡ 68b9c6dc-cd6b-4a0c-bc9f-73719ea229e7
let
	if false
		for i in 1:length(fr_bd_p[1])
			plt = fourrooms_heatmap_valuefunction(getindex.(fr_bd_joint_p, i))
			savefig(plt, "../fourrooms_example/joint_comp_values_$(i).png")
		end
	end
end

# ╔═╡ 232b6313-9e54-4894-9afc-fb5618a2307b
# getindex.(fr_bd_joint_p, 5)

# ╔═╡ e8035c2e-7e64-4884-ba2c-afb2dae72fb3
fr_comp_bdemon_const_joint_p, fr_comp_bdemon_const_joint = let	
	env_size = size(CompGVFs.FourRooms())
	env_feat_size = env_size[1] * env_size[2] 
	bdemon = CompGVFs.BDemon(
		env_feat_size, 
		4, 
		CompGVFs.GVFCumulant(fr_bd_joint_hrd[4]),
		# FeatureCumulant(11),
		CompGVFs.ϵGreedy(0.1),
		CompGVFs.ConstantDiscount(0.9))
	
	CompGVFs.fourrooms_behavior_offpolicy!(
		bdemon, 1_000_000, CompGVFs.Qλ(0.01, 0.9)), bdemon
end

# ╔═╡ 39f2f12e-ae2a-4139-8b6d-5ddbb94663d9
plot(CompGVFs.FourRooms(), fr_comp_bdemon_const_joint)

# ╔═╡ 7a08e753-5b71-46a5-bf64-3f135ba5b692
let
	if false
		plt = plot(CompGVFs.FourRooms(), fr_comp_bdemon_const_joint)
		savefig(plt, "../fourrooms_example/joint_comp_bdemon_const.png")
		plt = fourrooms_heatmap_policy(fr_comp_bdemon_const_joint_p)
		savefig(plt, "../fourrooms_example/joint_comp_bdemon_const_heatmap.png")
	end
end

# ╔═╡ 25b766eb-c465-4628-a8f4-b24c7e18a7ba
fr_neg_bdemon_const_joint_p, fr_neg_bdemon_const_joint = let	
	env_size = size(CompGVFs.FourRooms())
	env_feat_size = env_size[1] * env_size[2] 
	bdemon = CompGVFs.BDemon(
		env_feat_size, 
		4, 
		NegateCumulant(
			CompGVFs.GVFCumulant(fr_bd_joint_hrd[1])),
		# FeatureCumulant(11),
		CompGVFs.ϵGreedy(0.1),
		CompGVFs.ConstantDiscount(0.9))
	
	CompGVFs.fourrooms_behavior_offpolicy!(
		bdemon, 1_000_000, CompGVFs.Qλ(0.01, 0.9)), bdemon
end

# ╔═╡ fa7fe4a9-0619-4dda-8de7-8b5bacb9757e
plot(CompGVFs.FourRooms(), fr_neg_bdemon_const_joint)

# ╔═╡ 0e481bf5-8ef7-449a-9d55-452697c46c57
fourrooms_heatmap_policy(fr_neg_bdemon_const_joint_p)

# ╔═╡ e499afc3-8894-4532-a5cf-d139a04d90f1
fourrooms_heatmap_policy(fr_comp_bdemon_const_joint_p)

# ╔═╡ 9738a622-d358-4766-8d3c-2b1743c8287e
fr_neg_bdemon_p, fr_neg_bdemon = let	
	env_size = size(CompGVFs.FourRooms())
	env_feat_size = env_size[1] * env_size[2] 
	bdemon = CompGVFs.BDemon(
		env_feat_size, 
		4, 
		NegateCumulant(
			CompGVFs.FeatureCumulant(11)),
		# FeatureCumulant(11),
		CompGVFs.ϵGreedy(0.1),
		CompGVFs.TerminatingDiscount(0.9, 11))
	
	CompGVFs.fourrooms_behavior_offpolicy!(
		bdemon, 1_000_000, CompGVFs.Qλ(0.01, 0.9)), bdemon
end

# ╔═╡ 8185a4e2-e31d-4be8-8686-c1be494cf6c0
fourrooms_heatmap_policy(fr_neg_bdemon_p)

# ╔═╡ ee84b679-cb40-4d6b-bc9c-fb8c1bdb1943
plot(CompGVFs.FourRooms(), fr_neg_bdemon)

# ╔═╡ Cell order:
# ╠═903d11ae-186b-4dcc-bab5-14d9b1455069
# ╠═caaec4e0-67ba-4a38-9815-98faa726998d
# ╠═dacb15d9-7bbd-4195-b8b6-9f3053f1052b
# ╠═5a44cc9f-4e4c-492e-8aee-870fc4e52e6e
# ╠═c95ff355-9c70-410f-b852-f7c3004a773e
# ╠═d107cd3a-b441-4a42-8c62-90034bcb028c
# ╠═ce18fa61-a224-4230-a0cb-55d84b9d1295
# ╠═6fb15522-98e6-46a4-832c-432da771e8a9
# ╠═3dde5483-4e7f-44a2-a21e-801bfbe4b623
# ╠═84316175-c9ac-491e-ac2f-072e7fc5f93c
# ╠═9aa92437-6df6-4a95-8810-e295423b20b6
# ╠═d89eec44-5df2-4a46-b9db-8e68adbe6c3e
# ╠═206a7861-dca5-433f-b5af-f2c3449f97bc
# ╠═6bd6bbc9-b1f6-4a72-abbb-874bb1c64243
# ╠═fb9fd7af-c918-4632-b695-6396359fe480
# ╠═c792c013-ea1a-435b-8cb6-f284c754417d
# ╠═2fb9bfc2-1df2-4c96-b632-533cc41b463c
# ╠═6ad53250-6177-40f2-bdad-be3851294172
# ╠═cd3b5c29-6a2c-45b2-afc8-1aa59e96f519
# ╠═0037bb06-d6b5-45ae-84b9-31af35683d25
# ╠═2bb526c7-576f-421a-bc1c-0580d09dd9e8
# ╠═dc5b5f44-5e4a-4eda-9c6d-c378c79e553d
# ╠═e0f7130b-b26f-49cc-bea1-132afc4eee6d
# ╠═808e8e40-d60a-4f17-a605-5b6b287a1c18
# ╠═1092fdbc-8987-451f-a8d0-dcbe48a2f340
# ╠═87bfaf12-a965-48ff-b243-eae9902e6d6f
# ╠═7341e0f7-ab92-436e-ad77-2900122b60c3
# ╠═0db6b442-f0e9-4057-9b6b-0db16ac250eb
# ╠═967863c7-b4b7-48d7-a811-42e8c3f1f62f
# ╠═49c6eb84-7bf5-4cd1-aa64-b0a34abf4af4
# ╠═abd2f153-9aeb-4aa6-a368-c561b6f27701
# ╠═604bc156-047c-47f7-9c69-4d7dc0fc9692
# ╠═8e93d983-143c-4d98-9702-e2ff956a425d
# ╠═4dd55820-2f55-4cd4-bc14-54ad6a0418b5
# ╠═68b9c6dc-cd6b-4a0c-bc9f-73719ea229e7
# ╠═232b6313-9e54-4894-9afc-fb5618a2307b
# ╠═e8035c2e-7e64-4884-ba2c-afb2dae72fb3
# ╠═39f2f12e-ae2a-4139-8b6d-5ddbb94663d9
# ╠═7a08e753-5b71-46a5-bf64-3f135ba5b692
# ╠═25b766eb-c465-4628-a8f4-b24c7e18a7ba
# ╠═fa7fe4a9-0619-4dda-8de7-8b5bacb9757e
# ╠═0e481bf5-8ef7-449a-9d55-452697c46c57
# ╠═e499afc3-8894-4532-a5cf-d139a04d90f1
# ╠═9738a622-d358-4766-8d3c-2b1743c8287e
# ╠═8185a4e2-e31d-4be8-8686-c1be494cf6c0
# ╠═ee84b679-cb40-4d6b-bc9c-fb8c1bdb1943
