#!/usr/bin/env python3
"""Record Laya's language/script detector over a broad corpus (#339).

neuron routes each decision request to the English or multilingual Laya
checkpoint with a port of `laya.lang.analyse`. That function is a pile of
heuristics sitting on Python's Unicode semantics -- `str.isalpha`, `re`'s
`\\w`, `unicodedata.combining`, `str.lower`, `str.split` -- and a port that
reads the same can still route differently. This records the reference's
answer over a corpus built to reach every rule and every place those
semantics could diverge:

  - every script range, real sentences and generated letter runs
  - each function-word language, accented and ASCII-stripped, long and short
  - English with loanwords, names, symbols and IPA; code, URLs, e-mails,
    identifiers, acronyms and slash compounds
  - mixed-script and mixed-language states, multi-line and structured
  - the 4000-character budgets, the nesting depth limit, non-string leaves
  - combining marks, decomposed text, case-mapping oddities (İ, ß, final
    sigma, titlecase digraphs), fullwidth forms, numerals, emoji, and the
    whitespace Python and Rust disagree about
  - a seeded fuzz section mixing all of the above

Every mapping in the corpus has its keys in sorted order, so a consumer
whose JSON maps iterate in key order (serde_json without `preserve_order`)
sees the same leaf order Python does.

Characters are limited to those assigned in the Unicode version of the
Python that records the corpus, so a newer Unicode table on the consuming
side cannot disagree about a code point Python considers unassigned.

Usage (any Python 3.10+ with the laya SDK importable; no model, no network):

    python script/laya-lang-reference.py \\
        --out crates/neuron/src/harness/testdata/laya/lang_corpus.json
"""
import argparse
import json
import random
import sys
import unicodedata

import laya
from laya.lang import _WORD, _iter_text, analyse

SENTENCES = {
    "en": "Please refund the duplicate charge, I was billed twice this month and it is not fair.",
    "en_short": "refund please",
    "en_loanword": "I would like a café table and my résumé is attached for the role.",
    "en_name": "José and Zoë have the tickets that were sent to you from the office.",
    "en_two_loanwords": "The naïve café owner sent the résumé to you and it was not what we wanted.",
    "en_symbols": "Set α to 0.05 and β to 0.9, then run the job again with the new values.",
    "en_russian_name": "The report from Дмитрий Петрович Савицкий was sent to the team yesterday.",
    "en_ipa": "Vladimir is pronounced [vlɐˈdʲimʲɪr] in Russian and that is how we say it.",
    "en_stress": "The name Влади́мир has a stress mark on the second syllable for you.",
    "fr": "Bonjour, je voudrais annuler ma commande car elle est arrivée très en retard.",
    "fr_short": "merci beaucoup",
    "de": "Die Rechnung ist falsch, bitte korrigieren Sie den Betrag für diesen Monat.",
    "de_short": "Danke schön",
    "de_eszett": "Ich möchte die Straße und die Größe meiner Bestellung ändern, bitte.",
    "es": "Me han cobrado dos veces este mes y quiero que me devuelvan el dinero ya.",
    "pt": "Não consigo acessar a minha conta, já tentei redefinir a senha várias vezes.",
    "pt_br": "Voce pode me mandar a nota fiscal? Nao recebi nada ate agora.",
    "pt_ticket": "Deu erro 500 no endpoint de login depois do update de ontem.",
    "it": "Vorrei sapere quando arriverà il mio ordine, è già in ritardo di una settimana.",
    "nl": "Het pakket is niet aangekomen en ik wil graag een terugbetaling voor deze bestelling.",
    "ro": "Vreau să anulez comanda pentru că produsul este deteriorat și nu funcționează.",
    "bn": "ami taka ferot chai, amar order ekhono ashe nai, ki korbo bolun",
    "az": "Mən sifarişi ləğv etmək istəyirəm, çünki məhsul çox gec gəldi.",
    "tr": "Siparişimi iptal etmek istiyorum çünkü ürün çok geç geldi ve kırık.",
    "pl": "Chciałbym zwrócić pieniądze, ponieważ paczka dotarła uszkodzona.",
    "cs": "Chtěl bych vrátit peníze, protože zásilka dorazila poškozená.",
    "hu": "Szeretném visszakapni a pénzemet, mert a csomag sérülten érkezett.",
    "vi": "Tôi muốn được hoàn tiền vì gói hàng đã đến bị hư hỏng.",
    "sv": "Jag vill ha pengarna tillbaka eftersom paketet kom fram skadat.",
    "da": "Jeg vil gerne have pengene tilbage, fordi pakken kom i stykker.",
    "sw": "Nataka kurudishiwa pesa zangu kwa sababu kifurushi kiliharibika.",
    "la_shared_only": "la la e e o o un un",
    "el": "Θέλω να ακυρώσω την παραγγελία μου γιατί ήρθε πολύ αργά.",
    "el_upper_sigma": "ΘΕΛΩ ΤΗΝ ΕΠΙΣΤΡΟΦΗ ΧΡΗΜΑΤΩΝ ΤΟΥ ΛΟΓΑΡΙΑΣΜΟΥ ΣΑΣ",
    "ru": "Мне дважды списали деньги в этом месяце, верните, пожалуйста.",
    "uk": "Мені двічі списали гроші цього місяця, поверніть, будь ласка.",
    "hy": "Ես ուզում եմ չեղարկել պատվերս, քանի որ այն ուշ հասավ։",
    "he": "אני רוצה לבטל את ההזמנה שלי כי היא הגיעה באיחור.",
    "he_niqqud": "שָׁלוֹם, אֲנִי רוֹצֶה לְבַטֵּל אֶת הַהַזְמָנָה",
    "ar": "تم خصم المبلغ مرتين هذا الشهر، أريد استرداد أموالي من فضلكم.",
    "fa": "می‌خواهم سفارشم را لغو کنم چون خیلی دیر رسید.",
    "hi": "मुझे इस महीने दो बार शुल्क लिया गया, मुझे पैसे वापस चाहिए।",
    "bn_script": "এই মাসে আমার কাছ থেকে দুবার টাকা কাটা হয়েছে, টাকা ফেরত চাই।",
    "pa": "ਮੈਨੂੰ ਇਸ ਮਹੀਨੇ ਦੋ ਵਾਰ ਚਾਰਜ ਕੀਤਾ ਗਿਆ, ਪੈਸੇ ਵਾਪਸ ਚਾਹੀਦੇ ਹਨ।",
    "gu": "મને આ મહિને બે વાર ચાર્જ કરવામાં આવ્યો, પૈસા પાછા જોઈએ છે.",
    "or": "ମୋତେ ଏହି ମାସରେ ଦୁଇଥର ଚାର୍ଜ କରାଯାଇଛି।",
    "ta": "இந்த மாதம் எனக்கு இரண்டு முறை கட்டணம் வசூலிக்கப்பட்டது.",
    "te": "ఈ నెలలో నాకు రెండుసార్లు ఛార్జ్ చేశారు, డబ్బు తిరిగి ఇవ్వండి.",
    "kn": "ಈ ತಿಂಗಳು ನನಗೆ ಎರಡು ಬಾರಿ ಶುಲ್ಕ ವಿಧಿಸಲಾಗಿದೆ.",
    "ml": "ഈ മാസം എനിക്ക് രണ്ടുതവണ ചാർജ് ഈടാക്കി.",
    "si": "මෙම මාසයේ මට දෙවරක් ගාස්තු අය කළා.",
    "th": "เดือนนี้ฉันถูกเรียกเก็บเงินสองครั้ง ขอเงินคืนด้วยค่ะ",
    "lo": "ເດືອນນີ້ຂ້ອຍຖືກເກັບເງິນສອງເທື່ອ",
    "bo": "ང་ལ་ཟླ་བ་འདིར་ཐེངས་གཉིས་རིན་བསྡུས་སོང་།",
    "my": "ဒီလမှာ ကျွန်တော့်ကို နှစ်ကြိမ် ငွေတောင်းခံခဲ့တယ်။",
    "ka": "ამ თვეში ორჯერ ჩამომეჭრა თანხა, გთხოვთ დამიბრუნოთ.",
    "am": "በዚህ ወር ሁለት ጊዜ ክፍያ ተቀንሶብኛል፣ እባክዎ ይመልሱልኝ።",
    "km": "ខ្ញុំត្រូវបានគិតប្រាក់ពីរដងក្នុងខែនេះ ខ្ញុំចង់បានប្រាក់វិញ",
    "ko": "이번 달에 두 번 청구되었습니다. 환불해 주세요.",
    "ja": "今月二回請求されました。返金をお願いします。",
    "ja_kana": "ありがとうございます。キャンセルしたいです。",
    "zh": "这个月我被扣了两次钱，我要退款，请尽快处理。",
    "zh_trad": "這個月我被扣了兩次錢，我要退款。",
    "yue_ext_b": "𠮷野家の𩸽を食べたい",
    "mn_cyr": "Би захиалгаа цуцлахыг хүсч байна.",
    "bopomofo": "ㄅㄆㄇㄈ ㄉㄊㄋㄌ",
    "halfwidth_kana": "ｷｬﾝｾﾙｼﾀｲﾃﾞｽ",
    "cherokee": "ᏣᎳᎩ ᎦᏬᏂᎯᏍᏗ",
}

EXTRA = [
    "", " ", "   \n\t  ", "12345 67890", "3.14 2.71 1.41", "!!! ??? ...", "ok", "a", "no",
    "🙂🙂🙂 👍", "refund 🙂 please 👍 thanks", "café", "résumé naïve café",
    "ß", "Straße", "İstanbul İzmir", "ıi İI", "ǅemal ǈubljana ǋ", "ⓐⓑⓒ ⒶⒷⒸ", "𝐀𝐁𝐂 𝐚𝐛𝐜 test",
    "Ⅰ Ⅱ Ⅲ ⅳ ⅴ", "x² y³ ½ ¾", "٣٤٥ ١٢", "१२३ ४५६", "ＦＵＬＬＷＩＤＴＨ ｔｅｘｔ ｈｅｒｅ",
    "ʰʲʷ ˈˌ", "ɐɑɒ ʃʒ θð", "ƀƁƂ ǝǞ ȀȁȂ", "Ȿ ɀ", "ḀḁḂḃ ẞ ỳỹ",
    "a\x1cb\x1dc\x1ed\x1ff", "word\u00a0word\u00a0word\u00a0word", "一\u3000二\u3000三",
    "tab\tseparated\tvalues\there", "zero\u200bwidth\u200bjoiner", "soft\u00adhyphen",
    "e\u0301 e\u0300 a\u0308 o\u0303", "Tôi muốn được hoàn tiền",
    unicodedata.normalize("NFD", "Tôi muốn được hoàn tiền vì gói hàng đã đến bị hư hỏng."),
    unicodedata.normalize("NFD", "Não consigo acessar a minha conta, já tentei várias vezes."),
    unicodedata.normalize("NFD", "Vreau să anulez comanda pentru că produsul este deteriorat."),
    "ΟΔΟΣ ΟΔΟΣ ΟΔΟΣ", "Σ", "ΑΣ ΑΣ", "ὈΔΥΣΣΕΎΣ",
]

CODE_AND_IDENTIFIERS = [
    "see https://github.com/acme/os.path and user@acme.com for details",
    "email me at joao.silva@empresa.com.br or maria@exemplo.pt please",
    "version v1.2.3 of the lib, see U.S.A. rules and e.g. the docs",
    "x = round(el, 2); return non_english",
    "import os.path\nfrom foo import bar",
    "def f(x):\n    return {'la': 1, 'le': 2}",
    "Nav/Com OS/2 C:\\DOS\\mode ESA/UN are names not sentences at all",
    "MON LA EST COM DES are teams, states and radio bands in the listing",
    "COM COM COM COM radio listing for the English region",
    "O CLIENTE ESTÁ MUITO IRRITADO COM O ATRASO DO PEDIDO",
    "call foo(bar) then baz(qux) and see what the output of the call is",
    "Deu erro (500) no login depois do update de ontem",
    "a.b.c.d e@f g.h-i j_k.l m-n.o",
    "-leading.dash and trailing.dot. and @handle and .hidden files here",
    "..... @@@@ .@. a.@b",
    "x" * 300 + ".com",
    "path/to/file.py line 42 in <module> raise ValueError('não funciona')",
]

MULTILINE = [
    "Hello team,\nPlease see below.\nNão consigo acessar a minha conta, já tentei redefinir a senha.\nThanks",
    "Traceback (most recent call last):\n  File \"app.py\", line 3, in <module>\n    x = foo()\nO sistema não "
    "está funcionando desde ontem e preciso de ajuda urgente com isso",
    "Line one in English is here.\nLa fattura è sbagliata, per favore correggetela subito.",
    "Short line\n\n\nAnother short\nDie Rechnung ist falsch und ich bitte um eine Korrektur heute.",
    "English only line one.\nEnglish only line two.\nAnd the third line of English text.",
    "José\nMaría\nFrançois\nplease refund the order that was sent to them",
    "Ticket #1\nこんにちは、注文をキャンセルしたいです\nThanks for the help with this",
    # Targeted: each of the next cases flips if one rule is ported wrong.
    # All-caps function words in mixed-case text are acronyms, not French.
    "Please refund the duplicate charge, I was billed twice this month and it is not fair.\n"
    "Scores: MON LA EST DES COM LES ET tonight",
    # U+001C splits tokens for str.split(), so the identifier is dropped
    # alone rather than taking the Portuguese words with it.
    "Please refund the duplicate charge, I was billed twice this month.\n"
    "Não\x1cconsigo\x1cacessar\x1cminha\x1cconta\x1cjá\x1cx.y",
    # A combining stress mark must not split a capitalised name into a
    # lowercase "word".
    "The customer name on the invoice is Влади́мир "
    "Петро́вич and we need it fixed today",
    # One loanword: English function words rescue the marginal diacritic rate.
    "We will meet at the café for you",
    "I think the naïve plan is fine for all of us here",
]


def budget_cases():
    en = "Please refund the duplicate charge on my card for this month. "
    pt = "Não consigo acessar a minha conta, já tentei redefinir a senha várias vezes. "
    out = []
    for n in (3990, 3999, 4000, 4001, 4010, 8000):
        out.append((en * (n // len(en) + 1))[:n])
    out.append((en * 70)[:3995] + " " + pt)
    out.append((en * 70) + pt)
    out.append((en * 100) + "\n" + pt)
    out.append("x" * 3999 + "\n" + pt)
    out.append("x" * 4000 + "\n" + pt)
    out.append(("é" * 5000))
    out.append(("ab " * 1400) + pt)
    return out


def structured_cases():
    en = SENTENCES["en"]
    pt = SENTENCES["pt"]
    de = SENTENCES["de"]
    ja = SENTENCES["ja"]
    deep = "Não consigo acessar a minha conta, já tentei redefinir a senha."
    nested = deep
    deeps = []
    for depth in range(9):
        nested = [nested]
        deeps.append(nested)
    return [
        {"body": pt, "subject": "Refund request"},
        {"body": en, "note": pt},
        {"a_note": en * 60, "b_message": de},
        {"a_note": en * 60, "b_message": "Hallo, bitte."},
        {"customer": "José", "message": en},
        {"name": "Дмитрий Петрович", "text": en},
        {"message": ja, "subject": "Order"},
        {"count": 3, "flag": True, "missing": None, "text": en},
        {"count": 3, "flag": False, "text": pt},
        {"items": [en, pt, {"deep": de}], "meta": {"lang": "?"}},
        {"conversation": [{"content": en, "role": "user"}, {"content": pt, "role": "user"}]},
        {"a": {"b": {"c": {"d": {"e": {"f": {"g": pt}}}}}}},
        {"a": {"b": {"c": {"d": {"e": {"f": pt}}}}}},
        {"code": "x = round(el, 2)\nreturn os.path", "text": en},
        {"body": "Traceback:\n  File x.py\nO sistema não está funcionando desde ontem e preciso de ajuda", "id": 7},
        {"a": "José", "b": "María", "c": "François", "d": en},
        {"a": en * 70, "b": "Vreau să anulez comanda pentru că produsul este deteriorat."},
        {"a": en * 70, "b": "Мне дважды списали деньги в этом месяце, верните."},
        {"a": "ok", "b": "no", "c": "yes"},
        {},
        [],
        [en, pt],
        [pt, en],
        [en, en, en, "Die Rechnung ist falsch, bitte korrigieren."],
        ["hello", "bonjour", "hola"],
        [1, 2, 3],
        [None, True, 2.5],
        [[["deep", ["deeper", [pt]]]]],
        *deeps,
        12345,
        3.5,
        True,
        None,
    ]


def fuzz_cases(n, seed):
    rng = random.Random(seed)
    pools = [v for v in SENTENCES.values()] + EXTRA + CODE_AND_IDENTIFIERS
    words = []
    for s in pools:
        words.extend(s.split())
    specials = list("\n\t .,;:=()[]{}/\\@_-'\"") + ["\x1c", "\u00a0", "\u3000", "\u0301", "\u0308",
                                                   "\u200b", "İ", "ı", "ß", "Σ", "ǅ", "Ⅲ", "²",
                                                   "🙂", "１", "Ａ", "ａ", "ʰ", "ɐ", "é", "ñ"]
    out = []
    for _ in range(n):
        k = rng.randint(1, 40)
        parts = []
        for _ in range(k):
            r = rng.random()
            if r < 0.75:
                parts.append(rng.choice(words))
            else:
                parts.append(rng.choice(specials))
        sep = rng.choice([" ", " ", " ", "\n", ""])
        out.append(sep.join(parts))
    return out


def generated_script_runs():
    """A run of assigned letters from every script range the detector names."""
    from laya.lang import _SCRIPT_RANGES
    out = []
    for name, ranges in _SCRIPT_RANGES:
        for lo, hi in ranges:
            letters = [chr(cp) for cp in range(lo, hi + 1)
                       if unicodedata.category(chr(cp)) != "Cn"]
            if not letters:
                continue
            step = max(1, len(letters) // 24)
            sample = letters[::step][:24]
            out.append("".join(sample))
            out.append(" ".join("".join(sample[i:i + 4]) for i in range(0, len(sample), 4)))
    return out


def assigned(s):
    return all(unicodedata.category(c) != "Cn" and not (0xD800 <= ord(c) <= 0xDFFF) for c in s)


def sorted_keys(v):
    if isinstance(v, dict):
        return list(v) == sorted(v) and all(sorted_keys(x) for x in v.values())
    if isinstance(v, list):
        return all(sorted_keys(x) for x in v)
    return True


def all_strings(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from all_strings(x)
    elif isinstance(v, list):
        for x in v:
            yield from all_strings(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fuzz", type=int, default=400)
    ap.add_argument("--seed", type=int, default=339)
    args = ap.parse_args()

    states = []
    for name, s in SENTENCES.items():
        states.append(s)
        ascii_s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
        if ascii_s != s:
            states.append(ascii_s)
        states.append(s.upper())
        states.append(" ".join(s.split()[:3]))
    states += EXTRA + CODE_AND_IDENTIFIERS + MULTILINE + budget_cases()
    states += generated_script_runs()
    states += structured_cases()
    # Pairwise mixes: one English and one foreign line, both orders.
    for name, s in SENTENCES.items():
        if not name.startswith("en"):
            states.append(SENTENCES["en"] + "\n" + s)
            states.append(s + " " + SENTENCES["en"])
    states += fuzz_cases(args.fuzz, args.seed)

    cases, seen = [], set()
    for st in states:
        key = json.dumps(st, ensure_ascii=False, sort_keys=False)
        if key in seen:
            continue
        seen.add(key)
        assert sorted_keys(st), "mapping keys must be sorted: %r" % (st,)
        if not all(assigned(s) for s in all_strings(st)):
            continue
        det = analyse(st)
        # Recorded separately because a JSON reader that does not keep key
        # order loses the profile's order, which the reference defines.
        rec = {"state": st, "leaves": _iter_text(st), "analysis": det,
               "profile_order": list(det["script_profile"])}
        if isinstance(st, str):
            rec["words"] = _WORD.findall(st)
        cases.append(rec)

    header = {
        "laya_version": laya.__version__,
        "python": sys.version.split()[0],
        "unidata_version": unicodedata.unidata_version,
        "fuzz_seed": args.seed,
    }
    with open(args.out, "w") as f:
        json.dump({"header": header, "cases": cases}, f, ensure_ascii=False, indent=0)
    print("%d cases" % len(cases), file=sys.stderr)


if __name__ == "__main__":
    main()
