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


struct LinearCollection{L<:Linear}
    funcs::Vector{L}
end

function LinearCollection(in, out, num_funcs; init=(size...)->zeros(Float32, size...))
    LinearCollection([Linear(in, out; init=init) for i in 1:num_funcs])
end

function LinearCollection(in, num_funcs; init=(size...)->zeros(Float32, size...))
    LinearCollection([Linear(in; init=init) for i in 1:num_funcs])
end

predict(fa::LinearCollection, args...) = [predict(f, args...) for f in fa.funcs]


