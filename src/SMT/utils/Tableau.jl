struct Tableau
    size::Int64
    m::Float64

    #Figure out how to compute the sizes
    rows::Array{Entry,2}(undef, 2, 3)
    columns::Array{Entry,2}(undef, 2, 3)
    rowSize::Array{Int64}(undef, )
    columnSize::Array{Int64}(undef, )
    denseMap::Dict{UInt64, Entry}
end

function total_size(Tableau::Tableau)
	total = 0
    for s in Tableau.rowSize
        total += s
    end
    return total
end


# POTENTIALLY NOT NEEDED

function delete_all_entries(Tableau::Tableau)

end

function get_cell(Tableau::Tableau, row::UInt64, column::UInt64)
	Entry = rows[row]

    while row in Tableau.rowSize
        total += s
    end
    return total
end

function add_entry(Tableau::Tableau)
	total = 0
    for  s in Tableau.rowSize
        total += s
    end
    return total
end