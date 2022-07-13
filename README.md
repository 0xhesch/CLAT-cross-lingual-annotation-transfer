# CLAT-cross-lingual-annotation-transfer
This repository contains the source code for transferring annotations as described in 

> Cross-language transfer of high-quality annotations: Combining neural
> machine translation with cross-linguistic span alignment to apply NER
> to clinical texts in a low-resource language

Translating datasets with their annotations between multiple target languages is possible. The input format must be `conll`. 

The default selected model is `facebook/wmt21-dense-24-wide-en-x`, which means that the sourcedataset should be in English. The following 7 languages are supported by the model:

    English (en), Hausa (ha), Icelandic (is), Japanese (ja), Czech (cs), Russian (ru), Chinese (zh), German (de)

To translate the example dataset into German, run the following:

    >> python clat-fwd.py example.conll output.conll de

Example output should look like:

| Source Token | Source Entity | ->|Target Token|Target Entity
|--|--|--|--|--|
| Identification | O ||Identifikation|O|
| of | O ||von|O|
| APC2 | O ||ACP2|O|
| , | O ||,|O|
| a | O ||einem|O|
| homologue | O ||Homolgen|O|
| of | O ||des|O|
|the| O ||Tumorsuppressors|O|
|adenomatous|B-Disease||der|O|
|polyposis|I-Disease||adenomatösen|B-Disease|
|coli|I-Disease||Polyposis|I-Disease|
|tumour|I-Disease||coli|I-Disease|
|suppressor|O||.|O|
|.|O|


If there are bidirectional models for your language pair, that might be preferable to a large multilingual model. For other target languages, the `facebook/mbart-large-50-one-to-many-mmt` can be used instead, which covers the following 49 target languages

    Arabic (ar_AR), Czech (cs_CZ), German (de_DE), English (en_XX), Spanish (es_XX), Estonian (et_EE), Finnish (fi_FI), French (fr_XX), Gujarati (gu_IN), Hindi (hi_IN), Italian (it_IT), Japanese (ja_XX), Kazakh (kk_KZ), Korean (ko_KR), Lithuanian (lt_LT), Latvian (lv_LV), Burmese (my_MM), Nepali (ne_NP), Dutch (nl_XX), Romanian (ro_RO), Russian (ru_RU), Sinhala (si_LK), Turkish (tr_TR), Vietnamese (vi_VN), Chinese (zh_CN), Afrikaans (af_ZA), Azerbaijani (az_AZ), Bengali (bn_IN), Persian (fa_IR), Hebrew (he_IL), Croatian (hr_HR), Indonesian (id_ID), Georgian (ka_GE), Khmer (km_KH), Macedonian (mk_MK), Malayalam (ml_IN), Mongolian (mn_MN), Marathi (mr_IN), Polish (pl_PL), Pashto (ps_AF), Portuguese (pt_XX), Swedish (sv_SE), Swahili (sw_KE), Tamil (ta_IN), Telugu (te_IN), Thai (th_TH), Tagalog (tl_XX), Ukrainian (uk_UA), Urdu (ur_PK), Xhosa (xh_ZA), Galician (gl_ES), Slovene (sl_SI)

If the dataset is not in English, support languages can be used in between, there is also the `facebook/mbart-large-50-many-to-many-mmt` model.

