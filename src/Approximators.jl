

struct Linear{W<:AbstractArray}
    W::W
end

Linear(feat_size::Int; init=(size)->zeros(Float32, size)) = Linear(init(feat_size))
Tabular(size::Int; init=(sze)->zeros(Float32, sze)) = Linear(init(size))

# default
predict(fa::Linear{<:AbstractVector}, x) = dot(fa.w, x)

# Tile coded features
predict(fa::Linear{<:AbstractVector}, x::AbstractVector{Int}) = begin; 
    w = fa.w;
    ret = sum(view(w, x))
end

# Tabular
predict(fa::Linear{<:AbstractVector}, x::Int) = begin; 
    fa.w[x]
end


