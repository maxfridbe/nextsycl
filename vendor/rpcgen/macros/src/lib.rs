//! rpcgen's attributes (the runtime is the `rpcgen` crate). A method is its handler: what it is called, what it says it does, what it takes and what
//! it answers are read from the handler and its types, and everything else - the route, the OpenAPI document, the
//! API page, the web UI's TypeScript - is made from that (cp_core::rpc). Nothing is written twice, so nothing can
//! drift; what would drift is a compile error here instead:
//!
//! - `#[derive(Rpc)]` on a struct or a unit enum a method takes or answers: its shape as TypeScript and JSON
//!   Schema, following its serde attributes (`default`, `rename`, `rename_all`, `skip`, `flatten`, Option).
//!   A field may say more, as C#'s [Range] / [DefaultValue] do: `#[schema(min = 0.05, max = 1, default = 0.6)]`,
//!   `max_length`, `pattern`, `format`, `example`, or `json = r#"{...}"#` (any JSON Schema keywords) - all in the
//!   OpenAPI document. A field of a fixed set of values is a unit enum with #[derive(Rpc)] (a string union in
//!   TypeScript, an enum in the schema); `one_of` on a string does not compile.
//! - `#[rpc(name = "GetJob", tags = "jobs")]` on an `async fn(state, request) -> Result<Response, Error>`: the
//!   method. The doc comment is its summary (first line) and description. The name is PascalCase and starts
//!   with a verb; the request and the response must be `Rpc` types (else the trait bound fails to compile).
//!   `produces = "audio/wav, audio/mpeg"` names a download's media types (a Raw or Blob answer).
//! - `rpc_methods![State; a, b, jobs::c]`: a table of async methods (axum); `rpc_methods![sync State, Error; ...]`
//!   one of sync methods (`fn(&State, request) -> Result<Response, Error>`, Error: From<rpcgen::BadPayload>).
//!   Two methods of one name, or names of two styles (`GetJob` and `system.ping`), do not compile.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{parse_macro_input, Attribute, Data, DeriveInput, Error, Expr, Fields, FnArg, Ident, ItemFn, Lit, LitStr, Meta, Path, ReturnType, Token, Type};

/// The verbs a method name starts with (the RPC rule: named for what it does).
const VERBS: &[&str] = &[
    "Get", "List", "Set", "Start", "Stop", "Cancel", "Preview", "Create", "Generate", "Update", "Delete", "Upload", "Fetch",
    "Retry", "Add", "Remove", "Approve", "Skip", "Rerun", "Save", "Download", "Restore", "Pick", "Render", "Test", "Pull",
    "Reset", "Rename", "Clone", "Import", "Export", "Inspect", "Assign", "Split", "Slice", "Cut", "Vectorize", "Transcribe",
    "Design", "Prepare", "Rewrite", "Plan", "Refresh", "Regenerate", "Troubleshoot", "Restart", "Job", "Rpc", "Check",
    "Calibrate", "Analyze", "Inspect", "Merge", "Move", "Use", "Mark", "Open", "Run", "Send", "Write", "Read", "Find", "Search",
    "Paste", "Edit", "Recast", "Renarrate", "Remeasure", "Assemble", "Describe", "Replace", "Upscale", "Apply", "Choose",
];

fn doc_lines(attrs: &[Attribute]) -> Vec<String> {
    attrs
        .iter()
        .filter(|a| a.path().is_ident("doc"))
        .filter_map(|a| match &a.meta {
            Meta::NameValue(nv) => match &nv.value {
                Expr::Lit(l) => match &l.lit {
                    Lit::Str(s) => Some(s.value().trim().to_string()),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        })
        .collect()
}

/// The serde attributes that change a field's or a type's shape on the wire.
#[derive(Default)]
struct Serde {
    rename: Option<String>,
    rename_all: Option<String>,
    default: bool,
    skip: bool,
    skip_serializing_if: bool,
    flatten: bool,
    untagged_or_tagged: bool,
}

fn serde_of(attrs: &[Attribute]) -> syn::Result<Serde> {
    let mut s = Serde::default();
    for a in attrs.iter().filter(|a| a.path().is_ident("serde")) {
        a.parse_nested_meta(|m| {
            let key = m.path.get_ident().map(|i| i.to_string()).unwrap_or_default();
            match key.as_str() {
                "rename" => {
                    if m.input.peek(Token![=]) {
                        s.rename = Some(m.value()?.parse::<LitStr>()?.value());
                    } else {
                        m.parse_nested_meta(|n| {
                            if n.path.is_ident("deserialize") {
                                s.rename = Some(n.value()?.parse::<LitStr>()?.value());
                            } else {
                                let _ = n.value()?.parse::<LitStr>()?;
                            }
                            Ok(())
                        })?;
                    }
                }
                "rename_all" => s.rename_all = Some(m.value()?.parse::<LitStr>()?.value()),
                "default" => {
                    s.default = true;
                    if m.input.peek(Token![=]) {
                        let _ = m.value()?.parse::<LitStr>()?;
                    }
                }
                // #[serde(other)]: the catch-all of an enum read back from storage - not a value a caller sends
                "skip" | "skip_deserializing" | "other" => s.skip = true,
                "skip_serializing_if" => {
                    s.skip_serializing_if = true;
                    let _ = m.value()?.parse::<LitStr>()?;
                }
                "flatten" => s.flatten = true,
                "tag" | "content" | "untagged" => {
                    s.untagged_or_tagged = true;
                    if m.input.peek(Token![=]) {
                        let _ = m.value()?.parse::<LitStr>()?;
                    }
                }
                _ => {
                    // with, deserialize_with, alias, bound, deny_unknown_fields, ...: no change to the shape
                    if m.input.peek(Token![=]) {
                        let _ = m.value()?.parse::<Expr>()?;
                    } else if m.input.peek(syn::token::Paren) {
                        let content;
                        syn::parenthesized!(content in m.input);
                        let _: TokenStream2 = content.parse()?;
                    }
                }
            }
            Ok(())
        })?;
    }
    Ok(s)
}

fn renamed(name: &str, rule: Option<&str>) -> String {
    let words: Vec<String> = {
        let mut w = Vec::new();
        let mut cur = String::new();
        for (i, c) in name.chars().enumerate() {
            if c == '_' {
                if !cur.is_empty() {
                    w.push(std::mem::take(&mut cur));
                }
            } else if c.is_uppercase() && i > 0 && !cur.is_empty() {
                w.push(std::mem::take(&mut cur));
                cur.push(c);
            } else {
                cur.push(c);
            }
        }
        if !cur.is_empty() {
            w.push(cur);
        }
        w
    };
    let lower: Vec<String> = words.iter().map(|w| w.to_lowercase()).collect();
    let cap = |w: &String| {
        let mut c = w.chars();
        c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
    };
    match rule {
        Some("lowercase") => name.to_lowercase(),
        Some("UPPERCASE") => name.to_uppercase(),
        Some("snake_case") => lower.join("_"),
        Some("SCREAMING_SNAKE_CASE") => lower.join("_").to_uppercase(),
        Some("kebab-case") => lower.join("-"),
        Some("camelCase") => lower.iter().enumerate().map(|(i, w)| if i == 0 { w.clone() } else { cap(w) }).collect(),
        Some("PascalCase") => lower.iter().map(cap).collect(),
        _ => name.to_string(),
    }
}

/// A field's `#[schema(...)]`: the JSON Schema keywords it adds (as JSON text) and its string values (one_of).
fn schema_of(attrs: &[Attribute]) -> syn::Result<(String, Vec<String>)> {
    let mut kv: Vec<String> = Vec::new();
    let one_of: Vec<String> = Vec::new();
    let mut raw = String::new();
    let num = |e: &Expr| -> syn::Result<String> {
        match e {
            Expr::Lit(syn::ExprLit { lit: Lit::Int(i), .. }) => Ok(i.base10_digits().to_string()),
            Expr::Lit(syn::ExprLit { lit: Lit::Float(f), .. }) => Ok(f.base10_digits().to_string()),
            Expr::Unary(u) if matches!(u.op, syn::UnOp::Neg(_)) => match &*u.expr {
                Expr::Lit(syn::ExprLit { lit: Lit::Int(i), .. }) => Ok(format!("-{}", i.base10_digits())),
                Expr::Lit(syn::ExprLit { lit: Lit::Float(f), .. }) => Ok(format!("-{}", f.base10_digits())),
                _ => Err(Error::new_spanned(e, "a number")),
            },
            _ => Err(Error::new_spanned(e, "a number")),
        }
    };
    let json_lit = |e: &Expr| -> syn::Result<String> {
        match e {
            Expr::Lit(syn::ExprLit { lit: Lit::Str(s), .. }) => Ok(format!("{:?}", s.value())),
            Expr::Lit(syn::ExprLit { lit: Lit::Bool(b), .. }) => Ok(b.value.to_string()),
            _ => num(e),
        }
    };
    for a in attrs.iter().filter(|a| a.path().is_ident("schema")) {
        let metas = a.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)?;
        for m in metas {
            let Meta::NameValue(nv) = &m else { return Err(Error::new_spanned(m, "schema(key = value, ...)")) };
            let key = nv.path.get_ident().map(|i| i.to_string()).unwrap_or_default();
            let v = &nv.value;
            match key.as_str() {
                "min" => kv.push(format!("\"minimum\": {}", num(v)?)),
                "max" => kv.push(format!("\"maximum\": {}", num(v)?)),
                "max_length" => kv.push(format!("\"maxLength\": {}", num(v)?)),
                "min_length" => kv.push(format!("\"minLength\": {}", num(v)?)),
                "max_items" => kv.push(format!("\"maxItems\": {}", num(v)?)),
                "default" => kv.push(format!("\"default\": {}", json_lit(v)?)),
                "example" => kv.push(format!("\"example\": {}", json_lit(v)?)),
                "pattern" | "format" => kv.push(format!("{key:?}: {}", json_lit(v)?)),
                // a field of a fixed set of values is an enum, not a string with a list beside it: the handler matches
                // the enum, the TypeScript is its union, and no other value gets in (the user's rule, 2026-10-26)
                "one_of" => {
                    return Err(Error::new_spanned(
                        &nv.path,
                        "a field that takes one of a fixed set of values is an enum: a unit enum with #[derive(Rpc)] (serde's rename / rename_all for its words), not a String with one_of",
                    ));
                }
                "json" => {
                    let Expr::Lit(syn::ExprLit { lit: Lit::Str(s), .. }) = v else { return Err(Error::new_spanned(v, "json = r#\"{...}\"#")) };
                    let t = s.value();
                    let t = t.trim();
                    if !(t.starts_with('{') && t.ends_with('}')) {
                        return Err(Error::new_spanned(v, "json is a JSON object of schema keywords"));
                    }
                    raw = t[1..t.len() - 1].trim().to_string();
                }
                other => return Err(Error::new_spanned(&nv.path, format!("schema({other} = ...): min, max, default, example, one_of, max_length, min_length, max_items, pattern, format, json"))),
            }
        }
    }
    if !raw.is_empty() {
        kv.push(raw);
    }
    Ok((if kv.is_empty() { String::new() } else { format!("{{{}}}", kv.join(", ")) }, one_of))
}

/// `#[derive(Rpc)]`: the type's shape for the RPC documents and the TypeScript client.
#[proc_macro_derive(Rpc, attributes(schema))]
pub fn derive_rpc(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    derive(input).unwrap_or_else(|e| e.to_compile_error()).into()
}

fn derive(input: DeriveInput) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let name_s = name.to_string();
    let docs = doc_lines(&input.attrs);
    // a note about an internal route ("PUT /api/jobs/{id}/cover — ...") is the developer's, not the type's public
    // description: it stays in the source and out of the document
    let internal = docs.first().is_some_and(|l| {
        let l = l.trim_start_matches('`');
        ["GET ", "POST ", "PUT ", "DELETE ", "PATCH "].iter().any(|m| l.strip_prefix(m).is_some_and(|r| r.trim_start_matches('`').starts_with("/api")))
    });
    let doc = if internal { String::new() } else { docs.join(" ") };
    let cont = serde_of(&input.attrs)?;
    if !input.generics.params.is_empty() {
        return Err(Error::new_spanned(&input.generics, "#[derive(Rpc)] takes no generic types: name the concrete type"));
    }
    let body = match &input.data {
        Data::Struct(s) => {
            let Fields::Named(fields) = &s.fields else {
                return Err(Error::new_spanned(name, "#[derive(Rpc)] needs named fields (a JSON object)"));
            };
            let mut parts = Vec::new();
            for f in &fields.named {
                let fs = serde_of(&f.attrs)?;
                if fs.skip {
                    continue;
                }
                let ty = &f.ty;
                if fs.flatten {
                    parts.push(quote! { __fields.flatten(<#ty as ::rpcgen::RpcType>::object(__defs)); });
                    continue;
                }
                let ident = f.ident.as_ref().unwrap().to_string();
                let ident = ident.strip_prefix("r#").unwrap_or(&ident).to_string();
                let wire = fs.rename.clone().unwrap_or_else(|| renamed(&ident, cont.rename_all.as_deref()));
                let fdoc = doc_lines(&f.attrs).join(" ");
                let has_default = fs.default || cont.default || fs.skip_serializing_if;
                let (extra, one_of) = schema_of(&f.attrs)?;
                parts.push(quote! {
                    __fields.field_with::<#ty>(__defs, #wire, #fdoc, #has_default, #extra, &[#(#one_of),*]);
                });
            }
            quote! {
                fn shape(__defs: &mut ::rpcgen::Defs) -> ::rpcgen::Shape {
                    if __defs.begin(#name_s) {
                        let __obj = <Self as ::rpcgen::RpcType>::object(__defs);
                        __defs.named(#name_s, #doc, __obj);
                    }
                    ::rpcgen::Shape::Ref(#name_s.to_string())
                }
                fn object(__defs: &mut ::rpcgen::Defs) -> ::rpcgen::Object {
                    let mut __fields = ::rpcgen::Object::default();
                    #(#parts)*
                    __fields
                }
            }
        }
        Data::Enum(e) => {
            if cont.untagged_or_tagged {
                return Err(Error::new_spanned(name, "#[derive(Rpc)] on an enum: only unit variants (a string union) for now"));
            }
            let mut names = Vec::new();
            for v in &e.variants {
                if !matches!(v.fields, Fields::Unit) {
                    return Err(Error::new_spanned(v, "#[derive(Rpc)] on an enum: only unit variants (a string union) for now"));
                }
                let vs = serde_of(&v.attrs)?;
                if vs.skip {
                    continue;
                }
                names.push(vs.rename.unwrap_or_else(|| renamed(&v.ident.to_string(), cont.rename_all.as_deref())));
            }
            quote! {
                fn shape(__defs: &mut ::rpcgen::Defs) -> ::rpcgen::Shape {
                    if __defs.begin(#name_s) {
                        __defs.named_union(#name_s, #doc, &[#(#names),*]);
                    }
                    ::rpcgen::Shape::Ref(#name_s.to_string())
                }
            }
        }
        Data::Union(_) => return Err(Error::new_spanned(name, "#[derive(Rpc)]: a union has no JSON shape")),
    };
    Ok(quote! {
        impl ::rpcgen::RpcType for #name {
            #body
        }
    })
}

/// The arguments of `#[rpc(...)]`.
struct RpcArgs {
    name: Option<LitStr>,
    tags: Vec<String>,
    example: Option<LitStr>,
    examples: Option<LitStr>,
    /// a download's media types, as `produces = "audio/wav, audio/mpeg"`
    produces: Vec<String>,
}

impl Parse for RpcArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut a = RpcArgs { name: None, tags: Vec::new(), example: None, examples: None, produces: Vec::new() };
        let metas = Punctuated::<Meta, Token![,]>::parse_terminated(input)?;
        for m in metas {
            let Meta::NameValue(nv) = &m else { return Err(Error::new_spanned(m, "expected name = \"...\" or tags = \"a, b\"")) };
            let Expr::Lit(syn::ExprLit { lit: Lit::Str(s), .. }) = &nv.value else {
                return Err(Error::new_spanned(&nv.value, "a string"));
            };
            if nv.path.is_ident("name") {
                a.name = Some(s.clone());
            } else if nv.path.is_ident("tags") {
                a.tags = s.value().split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect();
            } else if nv.path.is_ident("example") {
                let t = s.value();
                if !t.trim_start().starts_with('{') {
                    return Err(Error::new_spanned(s, "example is a JSON object: the body a caller sends"));
                }
                a.example = Some(s.clone());
            } else if nv.path.is_ident("examples") {
                if !s.value().trim_start().starts_with('{') {
                    return Err(Error::new_spanned(s, "examples is a JSON object: name -> the body a caller sends"));
                }
                a.examples = Some(s.clone());
            } else if nv.path.is_ident("produces") {
                a.produces = s.value().split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect();
                if a.produces.is_empty() || a.produces.iter().any(|t| !t.contains('/') || t.contains(' ')) {
                    return Err(Error::new_spanned(s, "produces is a download's media types: \"audio/wav, audio/mpeg\""));
                }
            } else {
                return Err(Error::new_spanned(&nv.path, "#[rpc] takes name, tags, example, examples and produces"));
            }
        }
        Ok(a)
    }
}

/// `#[rpc(name = "GetJob", tags = "jobs")]` on a method's handler.
#[proc_macro_attribute]
pub fn rpc(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as RpcArgs);
    let f = parse_macro_input!(item as ItemFn);
    method(args, f).unwrap_or_else(|e| e.to_compile_error()).into()
}

fn method(args: RpcArgs, f: ItemFn) -> syn::Result<TokenStream2> {
    let fname = &f.sig.ident;
    let name = args.name.ok_or_else(|| Error::new(Span::call_site(), "#[rpc] needs name = \"<Method>\""))?;
    let n = name.value();
    let pascal = n.starts_with(|c: char| c.is_ascii_uppercase()) && n.chars().all(|c| c.is_ascii_alphanumeric());
    // dotted: segments that start lower case (system.ping, system.configSet, llm.describePromptGuideMD)
    let dotted = !n.is_empty()
        && n.split('.').count() > 1
        && n.split('.').all(|p| p.starts_with(|c: char| c.is_ascii_lowercase()) && p.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'));
    if !pascal && !dotted {
        return Err(Error::new_spanned(&name, "a method name is one PascalCase word (GetJob) or dotted, each part starting lower case (system.ping)"));
    }
    if pascal && !VERBS.iter().any(|v| n.starts_with(v) && n[v.len()..].chars().next().is_none_or(|c| c.is_ascii_uppercase())) {
        return Err(Error::new_spanned(&name, "a method name starts with what it does (Get, List, Set, Create, ...)"));
    }
    let sync = f.sig.asyncness.is_none();
    let inputs: Vec<&FnArg> = f.sig.inputs.iter().collect();
    let ty_of = |a: &FnArg| match a {
        FnArg::Typed(p) => Ok((*p.ty).clone()),
        FnArg::Receiver(r) => Err(Error::new_spanned(r, "a #[rpc] handler is a free fn")),
    };
    // a third parameter is the request's headers (a download honouring Range, say): async handlers only
    let (state, req, headers) = match inputs.as_slice() {
        [s] => (ty_of(s)?, None, false),
        [s, r] => (ty_of(s)?, Some(ty_of(r)?), false),
        [s, r, _h] if !sync => (ty_of(s)?, Some(ty_of(r)?), true),
        _ => return Err(Error::new_spanned(&f.sig.inputs, "a #[rpc] handler takes (state), (state, request) or, async, (state, request, headers)")),
    };
    // the answer's type through ::rpcgen::RpcResult, so an alias of a Result is fine and anything that is not
    // a Result of an Rpc response and an HTTP error fails the trait bound
    let ReturnType::Type(_, ret) = &f.sig.output else {
        return Err(Error::new_spanned(&f.sig, "a #[rpc] handler answers Result<Response, Error>"));
    };
    let resp: Type = if sync { syn::parse_quote!(<#ret as ::rpcgen::SyncResult>::Ok) } else { syn::parse_quote!(<#ret as ::rpcgen::RpcResult>::Ok) };
    let docs = doc_lines(&f.attrs);
    let summary = docs.first().cloned().unwrap_or_default();
    if summary.is_empty() {
        return Err(Error::new_spanned(fname, "a #[rpc] handler has a doc comment: its first line is the method's summary"));
    }
    // paragraphs (a blank doc line) are kept; the lines of one paragraph are one line
    let description = docs.split(|l| l.is_empty()).map(|p| p.join(" ")).filter(|p| !p.is_empty()).collect::<Vec<_>>().join("\n\n");
    let tags = &args.tags;
    let example = match &args.example {
        Some(e) => quote! { Some(#e) },
        None => quote! { None },
    };
    let examples = match &args.examples {
        Some(e) => quote! { Some(#e) },
        None => quote! { None },
    };
    let produces = &args.produces;
    let info = format_ident!("__rpc_info_{}", fname);
    let handler = format_ident!("__rpc_handler_{}", fname);
    let check = format_ident!("__rpc_check_{}", fname);
    let const_name = format_ident!("__RPC_NAME_{}", fname);
    let req_ty = req.clone().unwrap_or_else(|| syn::parse_quote!(::rpcgen::NoParams));
    let wait = if sync { quote! {} } else { quote! { .await } };
    // a sync handler has no headers to drop
    let drop_headers = if sync { quote! {} } else { quote! { let _ = __headers; } };
    let call = if headers {
        quote! { #fname(__state, __req, __headers) #wait }
    } else if req.is_some() {
        quote! { { #drop_headers #fname(__state, __req) #wait } }
    } else {
        quote! { { #drop_headers let _ = __req; #fname(__state) #wait } }
    };
    let vis = &f.vis;
    let handler_fn = if sync {
        quote! {
            #[doc(hidden)]
            #vis fn #handler(__state: #state, __body: ::rpcgen::serde_json::Value) -> ::rpcgen::SyncReply<#ret> {
                let __req: #req_ty = ::rpcgen::decode_sync::<#req_ty, <#ret as ::rpcgen::SyncResult>::Err>(__body)?;
                ::rpcgen::SyncResult::into_value(#call)
            }
        }
    } else {
        quote! {
            #[doc(hidden)]
            #vis fn #handler(__state: #state, __body: ::rpcgen::serde_json::Value, __headers: ::rpcgen::HeaderMap) -> ::rpcgen::Reply {
                Box::pin(async move {
                    let __req: #req_ty = match ::rpcgen::decode(__body) {
                        Ok(r) => r,
                        Err(e) => return e,
                    };
                    ::rpcgen::RpcResult::into_reply(#call)
                })
            }
        }
    };
    Ok(quote! {
        #f

        #[doc(hidden)]
        #vis fn #check(__body: &::rpcgen::serde_json::Value) -> Result<(), ::rpcgen::BadPayload> {
            ::rpcgen::check::<#req_ty>(__body)
        }

        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        #vis const #const_name: &str = #name;

        #[doc(hidden)]
        #vis fn #info() -> ::rpcgen::MethodInfo {
            ::rpcgen::MethodInfo::new::<#req_ty, #resp>(#name, #summary, #description, &[#(#tags),*]).with_example(#example).with_examples(#examples).with_produces(&[#(#produces),*])
        }

        #handler_fn
    })
}

/// `rpc_methods![State; a, b, module::c]`
struct Table {
    state: Type,
    /// a sync table: its error type
    error: Option<Type>,
    fns: Vec<Path>,
}

impl Parse for Table {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let sync = input.peek(Ident) && input.fork().parse::<Ident>().is_ok_and(|i| i == "sync") && !input.peek2(Token![;]);
        if sync {
            input.parse::<Ident>()?;
        }
        let state: Type = input.parse()?;
        let error = if sync {
            input.parse::<Token![,]>()?;
            Some(input.parse::<Type>()?)
        } else {
            None
        };
        input.parse::<Token![;]>()?;
        let fns = Punctuated::<Path, Token![,]>::parse_terminated(input)?.into_iter().collect();
        Ok(Table { state, error, fns })
    }
}

fn with_prefix(p: &Path, prefix: &str, upper: bool) -> Path {
    let mut p = p.clone();
    let last = p.segments.last_mut().unwrap();
    let id = last.ident.to_string();
    last.ident = Ident::new(&format!("{prefix}{id}"), last.ident.span());
    let _ = upper;
    p
}

/// A maker's table of methods: `pub fn rpc_methods() -> Vec<Method<State>>`, the names checked distinct at
/// compile time.
#[proc_macro]
pub fn rpc_methods(input: TokenStream) -> TokenStream {
    let t = parse_macro_input!(input as Table);
    let state = &t.state;
    let infos: Vec<Path> = t.fns.iter().map(|p| with_prefix(p, "__rpc_info_", false)).collect();
    let handlers: Vec<Path> = t.fns.iter().map(|p| with_prefix(p, "__rpc_handler_", false)).collect();
    let checks: Vec<Path> = t.fns.iter().map(|p| with_prefix(p, "__rpc_check_", false)).collect();
    let names: Vec<Path> = t.fns.iter().map(|p| with_prefix(p, "__RPC_NAME_", true)).collect();
    let table = match &t.error {
        None => quote! {
            /// The table's RPC methods, from their #[rpc] handlers (rpcgen).
            pub fn rpc_methods() -> Vec<::rpcgen::Method<#state>> {
                vec![#(::rpcgen::Method { info: #infos(), handler: #handlers, check: #checks }),*]
            }
        },
        Some(err) => quote! {
            /// The table's RPC methods, from their #[rpc] handlers (rpcgen).
            pub fn rpc_methods() -> Vec<::rpcgen::SyncMethod<#state, #err>> {
                vec![#(::rpcgen::SyncMethod { info: #infos(), handler: #handlers, check: #checks }),*]
            }
        },
    };
    quote! {
        #table
        const _: () = assert!(::rpcgen::distinct(&[#(#names),*]), "two #[rpc] methods in this table share a name");
        const _: () = assert!(::rpcgen::one_style(&[#(#names),*]), "a table's method names are all of one style: GetJob, or system.ping");
    }
    .into()
}
