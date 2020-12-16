#!/usr/bin/env python

from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer

from base import BaseAlgo
from codec import codecs_manager
from util import df_util
from util.param_util import convert_params


class TFIDF(BaseAlgo):
    def handle_options(self, options):
        if (
            len(options.get('feature_variables', [])) != 1
            or len(options.get('target_variable', [])) > 0
        ):
            raise RuntimeError('Syntax error: You must specify exactly one field')

    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            ints=['max_features'],
            strs=['max_df', 'min_df', 'stop_words', 'analyzer', 'norm', 'token_pattern'],
            ranges=['ngram_range'],
        )

        for doc_freq, default_val in [('max_df', 1.0), ('min_df', 1)]:
            if doc_freq in out_params:
                # EAFP... convert max_df/min_df to float/int if it is a number.
                try:
                    float_val = float(out_params[doc_freq])
                    int_val = int(float_val)
                except ValueError:
                    raise RuntimeError(
                        'Syntax Error: {doc_freq} requires a numeric value, e.g. {doc_freq}=1.0'.format(
                            doc_freq=doc_freq
                        )
                    )
                if float_val == 1.0:
                    out_params[doc_freq] = default_val
                elif float_val == int_val:
                    out_params[doc_freq] = int_val
                else:
                    out_params[doc_freq] = float_val

        if 'ngram_range' in list(out_params.keys()):
            if len(out_params['ngram_range']) != 2:
                raise RuntimeError(
                    'Syntax Error: ngram_range requires a range, e.g. ngram_range=1-5'
                )

        # TODO: Maybe let the user know that we make this change.
        out_params.setdefault('max_features', 100)

        self.estimator = _TfidfVectorizer(**out_params)

    def fit(self, df, options):
        # Make a copy of data, to not alter original dataframe
        X = df.copy()

        # Make sure to turn off get_dummies
        X, _, self.columns = df_util.prepare_features(
            X=X,
            variables=self.feature_variables,
            get_dummies=False,
            mlspl_limits=options.get('mlspl_limits'),
        )

        X = X.values.ravel().astype('str')
        self.estimator.fit(X)

    def make_output_names(self, options):
        default_name = self.feature_variables[0] + '_tfidf'
        output_name = options.get('output_name', default_name)
        feature_names = self.estimator.get_feature_names()
        output_names = [
            output_name + '_' + str(index) + '_' + word
            for (index, word) in enumerate(feature_names)
        ]
        return output_names

    def apply(self, df, options):
        # Make a copy of data, to not alter original dataframe
        X = df.copy()

        # Make sure to turn off get_dummies
        X, nans, _ = df_util.prepare_features(
            X=X,
            variables=self.feature_variables,
            final_columns=self.columns,
            get_dummies=False,
            mlspl_limits=options.get('mlspl_limits'),
        )

        X = X.values.ravel().astype('str')
        y_hat = self.estimator.transform(X)

        # Convert the returned sparse matrix into array
        y_hat = y_hat.toarray()

        output_names = self.make_output_names(options)

        output = df_util.create_output_dataframe(
            y_hat=y_hat, output_names=output_names, nans=nans
        )

        df = df_util.merge_predictions(df, output)
        return df

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec

        codecs_manager.add_codec('algos.TFIDF', 'TFIDF', SimpleObjectCodec)
        codecs_manager.add_codec(
            'sklearn.feature_extraction.text', 'TfidfVectorizer', SimpleObjectCodec
        )
        codecs_manager.add_codec(
            'sklearn.feature_extraction.text', 'TfidfTransformer', SimpleObjectCodec
        )
        codecs_manager.add_codec('scipy.sparse.dia', 'dia_matrix', SimpleObjectCodec)
