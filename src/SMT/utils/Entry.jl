struct Entry
    nextInRow::Entry
    prevInRow::Entry
    nextInColumn::Entry
    prevInColumn::Entry
    row::UInt32
    column::UInt32
    value::Float64
end
