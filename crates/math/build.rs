use std::{env, fs, path::PathBuf};

use proc_macro2::{Literal, TokenStream};
use quote::quote;
use syn::File;

const MAX_N: usize = 256;

fn main() {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let dest = out_dir.join("unroll.rs");

    let mut arms = TokenStream::new();

    for n in 1..=MAX_N {
        let n_lit = Literal::usize_unsuffixed(n);
        let elems: Vec<_> = (0..n).map(Literal::usize_unsuffixed).collect();
        let elems_inclusive: Vec<_> = (0..=n).map(Literal::usize_unsuffixed).collect();

        arms.extend(quote! {
            ($var:ident in (.. #n_lit), $body:expr) => {
                {
                    #(
                        let $var = #elems;
                        $body;
                    )*
                }
            };
            ($var:ident in (..= #n_lit), $body:expr) => {
                {
                    #(
                        let $var = #elems_inclusive;
                        $body;
                    )*
                }
            };
            ($var:ident in [.. #n_lit], $body:expr) => {
                [
                    #(
                        {
                            let $var = #elems;
                            $body
                        },
                    )*
                ]
            };
            ($var:ident in [..= #n_lit], $body:expr) => {
                [
                    #(
                        {
                            let $var = #elems_inclusive;
                            $body
                        },
                    )*
                ]
            };
        });
    }

    let file_tokens = quote! {
        #[macro_export]
        macro_rules! unroll {
            #arms
        }
    };

    let file: File = syn::parse2(file_tokens).unwrap();
    let code = prettyplease::unparse(&file);

    fs::write(&dest, code).unwrap();

    println!("cargo:rerun-if-changed=build.rs");
}
