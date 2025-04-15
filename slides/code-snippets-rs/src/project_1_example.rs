fn main() {
    a.chunks_mut(split_length)
        .par_bridge()
        .for_each(bsort);

    a.chunks_mut(split_length)
        .for_each(bsort);
}
