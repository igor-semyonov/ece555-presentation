struct FakeCat {
    alive: bool,
    hungry: bool,
}

enum RealCat {
    Alive {
        hungry: bool,
    },
    Dead,
}

fn main() {
    let fake_cat = FakeCat {
        alive: false,
        hungry: true,
    };

    let real_cat1 = RealCat::Dead;
    let real_cat2 = RealCat::Alive {
        hungry: true,
    };
    let real_cat3 = RealCat::Dead{hungry: true};

    println!("{} {} {}", real_cat1, real_cat2, real_cat3);
}
