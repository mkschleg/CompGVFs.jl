struct Linear{W<:AbstractArray}
    w::W
end

Linear(in, out; init=(size...)->zeros(Float32, size...)) = Linear(init(out, in))
Linear(feat_size; init=(size)->zeros(Float32, size)) = Linear(init(feat_size))
Tabular(size; init=(sze)->zeros(Float32, sze)) = Linear(init(size))
Tabular(in, out; init=(sze...)->zeros(Float32, sze...)) = Linear(init(in, out))

# default
predict(fa::Linear{<:AbstractVector}, x) = dot(fa.w, x)
predict(fa::Linear{<:AbstractMatrix}, x) = fa.w*x

# Tile coded features
predict(fa::Linear{<:AbstractVector}, x::AbstractVector{Int}) = begin; 
    w = fa.w;
    ret = sum(view(w, x))
end

predict(fa::Linear{<:AbstractMatrix}, x::AbstractVector{Int}) = begin; 
    w = fa.w;
    ret = sum(view(w, :, x); dims=2)
end

# Tabular
predict(fa::Linear{<:AbstractVector}, x::Int) = begin; 
    fa.w[x]
end

predict(fa::Linear{<:AbstractMatrix}, x::Int) = begin; 
    fa.w[:, x]
end

