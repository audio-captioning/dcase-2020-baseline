#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pickle
from time import time
from typing import MutableMapping, MutableSequence,\
    Any, Union, List, Dict, Tuple

from torch import Tensor, no_grad, save as pt_save, \
    load as pt_load, randperm
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam
from torch.nn.functional import softmax
from loguru import logger

from tools import file_io, printing, csv_functions
from tools.argument_parsing import get_argument_parser
from tools.model import module_epoch_passing, get_model,\
    get_device
from data_handlers.clotho_loader import get_clotho_loader
from eval_metrics import evaluate_metrics


__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['method']


def _decode_outputs(predicted_outputs: MutableSequence[Tensor],
                    ground_truth_outputs: MutableSequence[Tensor],
                    indices_object: MutableSequence[str],
                    file_names: MutableSequence[Path],
                    eos_token: str,
                    print_to_console: bool) \
        -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Decodes predicted output to string.

    :param predicted_outputs: Predicted outputs.
    :type predicted_outputs: list[torch.Tensor]
    :param ground_truth_outputs: Ground truth outputs.
    :type ground_truth_outputs: list[torch.Tensor]
    :param indices_object: Object to map indices to text (words or chars).
    :type indices_object: list[str]
    :param file_names: List of ile names used.
    :type file_names: list[pathlib.Path]
    :param eos_token: End of sequence token to be used.
    :type eos_token: str
    :param print_to_console: Print captions to console?
    :type print_to_console: bool
    :return: Predicted and ground truth captions for scoring.
    :rtype: (list[dict[str, str]], list[dict[str, str]])
    """
    caption_logger = logger.bind(is_caption=True, indent=None)
    main_logger = logger.bind(is_caption=False, indent=0)
    caption_logger.info('Captions start')
    main_logger.info('Starting decoding of captions')
    text_sep = '-' * 100

    captions_pred: List[Dict] = []
    captions_gt: List[Dict] = []
    f_names: List[str] = []

    if print_to_console:
        main_logger.info(f'{text_sep}\n{text_sep}\n{text_sep}\n\n')

    for gt_words, b_predictions, f_name in zip(
            ground_truth_outputs, predicted_outputs, file_names):
        predicted_words = softmax(b_predictions, dim=-1).argmax(1)

        predicted_caption = [indices_object[i.item()]
                             for i in predicted_words]
        gt_caption = [indices_object[i.item()]
                      for i in gt_words]

        gt_caption = gt_caption[:gt_caption.index(eos_token)]
        try:
            predicted_caption = predicted_caption[
                                :predicted_caption.index(eos_token)]
        except ValueError:
            pass

        predicted_caption = ' '.join(predicted_caption)
        gt_caption = ' '.join(gt_caption)

        f_n = f_name.stem.split('.')[0]

        if f_n not in f_names:
            f_names.append(f_n)
            captions_pred.append({
                'file_name': f_n,
                'caption_predicted': predicted_caption})
            captions_gt.append({
                'file_name': f_n,
                'caption_1': gt_caption})
        else:
            for d_i, d in enumerate(captions_gt):
                if f_n == d['file_name']:
                    len_captions = len([i_c for i_c in d.keys()
                                        if i_c.startswith('caption_')]) + 1
                    d.update({f'caption_{len_captions}': gt_caption})
                    captions_gt[d_i] = d
                    break

        log_strings = [f'Captions for file {f_name.stem}: ',
                       f'\tPredicted caption: {predicted_caption}',
                       f'\tOriginal caption: {gt_caption}\n\n']

        [caption_logger.info(log_string)
         for log_string in log_strings]

        if print_to_console:
            [main_logger.info(log_string)
             for log_string in log_strings]

    if print_to_console:
        main_logger.info(f'{text_sep}\n{text_sep}\n{text_sep}\n\n')

    logger.bind(is_caption=False, indent=0).info(
        'Decoding of captions ended')

    return captions_pred, captions_gt


def _do_testing(model: Module,
                settings_data:  MutableMapping[str, Any],
                settings_io:  MutableMapping[str, Any],
                indices_list: MutableSequence[str]) \
        -> None:
    """Evaluation of an optimized model.

    :param model: Model to use.
    :type model: torch.nn.Module
    :param settings_data: Data settings to use.
    :type settings_data: dict
    :param indices_list: Sequence with the words of the captions.
    :type indices_list: list[str]
    """
    model.eval()
    logger_main = logger.bind(is_caption=False, indent=1)

    data_path_test = Path(
        settings_io['root_dirs']['data'],
        settings_io['dataset']['features_dirs']['output'],
        settings_io['dataset']['features_dirs']['test'])

    logger_main.info('Getting test data')
    test_data = get_clotho_loader(
        settings_io['dataset']['features_dirs']['test'],
        is_training=False,
        settings_data=settings_data,
        settings_io=settings_io)
    logger_main.info('Done')

    text_sep = '-' * 100
    starting_text = 'Starting testing on test data'

    logger_main.info(starting_text)
    logger.bind(is_caption=True, indent=0).info(
        f'{text_sep}\n{text_sep}\n{text_sep}\n\n')
    logger.bind(is_caption=True, indent=0).info(
        f'{starting_text}.\n\n')

    with no_grad():
        test_outputs = module_epoch_passing(
            data=test_data, module=model,
            objective=None, optimizer=None)

    captions_pred, _ = _decode_outputs(
        test_outputs[1],
        test_outputs[2],
        indices_object=indices_list,
        file_names=list(data_path_test.iterdir()),
        eos_token='<eos>',
        print_to_console=False)

    # clotho_file_{file_name} to {file_name}.wav
    for i, entry in enumerate(captions_pred):
        entry['file_name'] = entry['file_name']\
            .replace('clotho_file_', '') + '.wav'
        captions_pred[i] = entry

    submission_dir = Path().joinpath(
        settings_io['root_dirs']['outputs'],
        settings_io['submissions']['submissions_dir'])
    submission_dir.mkdir(parents=True, exist_ok=True)
    csv_functions.write_csv_file(
        captions_pred,
        settings_io['submissions']['submission_file'],
        submission_dir,
        add_timestamp=True)

    logger_main.info('Testing done')


def _do_evaluation(model: Module,
                   settings_data:  MutableMapping[str, Any],
                   settings_io:  MutableMapping[str, Any],
                   indices_list: MutableSequence[str]) \
        -> None:
    """Evaluation of an optimized model.

    :param model: Model to use.
    :type model: torch.nn.Module
    :param settings_data: Data settings to use.
    :type settings_data: dict
    :param indices_list: Sequence with the words of the captions.
    :type indices_list: list[str]
    """
    model.eval()
    logger_main = logger.bind(is_caption=False, indent=1)

    data_path_evaluation = Path(
        settings_io['root_dirs']['data'],
        settings_io['dataset']['features_dirs']['output'],
        settings_io['dataset']['features_dirs']['evaluation'])

    logger_main.info('Getting evaluation data')
    validation_data = get_clotho_loader(
        settings_io['dataset']['features_dirs']['evaluation'],
        is_training=False,
        settings_data=settings_data,
        settings_io=settings_io)
    logger_main.info('Done')

    text_sep = '-' * 100
    starting_text = 'Starting evaluation on evaluation data'

    logger_main.info(starting_text)
    logger.bind(is_caption=True, indent=0).info(
        f'{text_sep}\n{text_sep}\n{text_sep}\n\n')
    logger.bind(is_caption=True, indent=0).info(
        f'{starting_text}.\n\n')

    with no_grad():
        evaluation_outputs = module_epoch_passing(
            data=validation_data, module=model,
            objective=None, optimizer=None)

    captions_pred, captions_gt = _decode_outputs(
        evaluation_outputs[1],
        evaluation_outputs[2],
        indices_object=indices_list,
        file_names=sorted(list(data_path_evaluation.iterdir())),
        eos_token='<eos>',
        print_to_console=False)

    logger_main.info('Evaluation done')

    metrics = evaluate_metrics(captions_pred, captions_gt)

    for metric, values in metrics.items():
        logger_main.info(f'{metric:<7s}: {values["score"]:7.4f}')


def _do_training(model: Module,
                 settings_training:  MutableMapping[
                     str, Union[Any, MutableMapping[str, Any]]],
                 settings_data:  MutableMapping[
                     str, Union[Any, MutableMapping[str, Any]]],
                 settings_io:  MutableMapping[
                     str, Union[Any, MutableMapping[str, Any]]],
                 model_file_name: str,
                 model_dir: Path,
                 indices_list: MutableSequence[str]) \
        -> None:
    """Optimization of the model.

    :param model: Model to optimize.
    :type model: torch.nn.Module
    :param settings_training: Training settings to use.
    :type settings_training: dict
    :param settings_data: Training data settings to use.
    :type settings_data: dict
    :param settings_io: Data I/O settings to use.
    :type settings_io: dict
    :param model_file_name: File name of the model.
    :type model_file_name: str
    :param model_dir: Directory to serialize the model to.
    :type model_dir: pathlib.Path
    :param indices_list: A sequence with the words.
    :type indices_list: list[str]
    """
    # Initialize variables for the training process
    prv_training_loss = 1e8
    patience: int = settings_training['patience']
    loss_thr: float = settings_training['loss_thr']
    patience_counter = 0
    best_epoch = 0

    # Initialize logger
    logger_main = logger.bind(is_caption=False, indent=1)

    # Inform that we start getting the data
    logger_main.info('Getting training data')

    # Get training data and count the amount of batches
    training_data = get_clotho_loader(
        settings_io['dataset']['features_dirs']['development'],
        is_training=True,
        settings_data=settings_data,
        settings_io=settings_io)

    logger_main.info('Done')

    # Initialize loss and optimizer objects
    objective = CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(),
                     lr=settings_training['optimizer']['lr'])

    # Inform that we start training
    logger_main.info('Starting training')

    model.train()
    for epoch in range(settings_training['nb_epochs']):

        # Log starting time
        start_time = time()

        # Do a complete pass over our training data
        epoch_output = module_epoch_passing(
            data=training_data,
            module=model,
            objective=objective,
            optimizer=optimizer,
            grad_norm=settings_training['grad_norm']['norm'],
            grad_norm_val=settings_training['grad_norm']['value'])
        objective_output, output_y_hat, output_y, f_names = epoch_output

        # Get mean loss of training and print it with logger
        training_loss = objective_output.mean().item()

        logger_main.info(f'Epoch: {epoch:05d} -- '
                         f'Training loss: {training_loss:>7.4f} | '
                         f'Time: {time() - start_time:>5.3f}')

        # Check if we have to decode captions for the current epoch
        if divmod(epoch + 1,
                  settings_training['text_output_every_nb_epochs'])[-1] == 0:

            # Get the subset of files for decoding their captions
            sampling_indices = sorted(randperm(len(output_y_hat))
                                      [:settings_training['nb_examples_to_sample']]
                                      .tolist())

            # Do the decoding
            _decode_outputs(*zip(*[[output_y_hat[i], output_y[i]]
                                 for i in sampling_indices]),
                            indices_object=indices_list,
                            file_names=[Path(f_names[i_f_name])
                                        for i_f_name in sampling_indices],
                            eos_token='<eos>',
                            print_to_console=False)

        # Check improvement of loss
        if prv_training_loss - training_loss > loss_thr:
            # Log the current loss
            prv_training_loss = training_loss

            # Log the current epoch
            best_epoch = epoch

            # Serialize the model keeping the epoch
            pt_save(
                model.state_dict(),
                str(model_dir.joinpath(
                    f'epoch_{best_epoch:05d}_{model_file_name}')))

            # Zero out the patience
            patience_counter = 0

        else:

            # Increase patience counter
            patience_counter += 1

        # Serialize the model and optimizer.
        for pt_obj, save_str in zip([model, optimizer], ['', '_optimizer']):
            pt_save(
                pt_obj.state_dict(),
                str(model_dir.joinpath(
                    f'latest{save_str}_{model_file_name}')))

        # Check for stopping criteria
        if patience_counter >= patience:
            logger_main.info('No lower training loss for '
                             f'{patience_counter} epochs. '
                             'Training stops.')
            break

    # Inform that we are done
    logger_main.info('Training done')

    # Load best model
    model.load_state_dict(pt_load(
        str(model_dir.joinpath(
            f'epoch_{best_epoch:05d}_{model_file_name}'))))


def _get_nb_output_classes(settings: MutableMapping[str, Any]) \
        -> int:
    """Gets the amount of output classes.

    :param settings: Settings to use.
    :type settings: dict
    :return: Amount of output classes.
    :rtype: int
    """
    f_name_field = 'words_list_file_name' \
        if settings['data']['output_field_name'].startswith('words') \
        else 'characters_list_file_name'

    f_name = settings['data']['files'][f_name_field]
    path = Path(
        settings['data']['files']['root_dir'],
        settings['data']['files']['dataset_dir'],
        f_name)

    with path.open('rb') as f:
        return len(pickle.load(f))


def _load_indices_file(settings_files: MutableMapping[str, Any],
                       settings_data: MutableMapping[str, Any]) \
        -> MutableSequence[str]:
    """Loads and returns the indices file.

    :param settings_files: Settings of file i/o to be used.
    :type settings_files: dict
    :param settings_data: Settings of data to be used. .
    :type settings_data: dict
    :return: The indices file.
    :rtype: list[str]
    """
    path = Path(
        settings_files['root_dirs']['data'],
        settings_files['dataset']['pickle_files_dir'])
    p_field = 'words_list_file_name' \
        if settings_data['output_field_name'].startswith('words') \
        else 'characters_list_file_name'
    return file_io.load_pickle_file(
        path.joinpath(settings_files['dataset']['files'][p_field]))


def method(settings: MutableMapping[str, Any]) \
        -> None:
    """Baseline method.

    :param settings: Settings to be used.
    :type settings: dict
    """
    logger_main = logger.bind(is_caption=False, indent=0)
    logger_main.info('Bootstrapping method')
    pretty_printer = printing.get_pretty_printer()
    logger_inner = logger.bind(is_caption=False, indent=1)
    device, device_name = get_device(
        settings['dnn_training_settings']['training']['force_cpu'])

    model_dir = Path(
        settings['dirs_and_files']['root_dirs']['outputs'],
        settings['dirs_and_files']['model']['model_dir'])

    model_dir.mkdir(parents=True, exist_ok=True)

    model_file_name = f'{settings["dirs_and_files"]["model"]["checkpoint_model_name"]}'

    logger_inner.info(f'Process on {device_name}\n')

    logger_inner.info('Settings:\n'
                      f'{pretty_printer.pformat(settings)}\n')

    logger_inner.info('Loading indices file')
    indices_list = _load_indices_file(
        settings['dirs_and_files'],
        settings['dnn_training_settings']['data'])
    logger_inner.info('Done')

    model: Union[Module, None] = None

    logger_main.info('Bootstrapping done')

    if settings['workflow']['dnn_training']:
        logger_main.info('Doing training')
        logger_inner.info('Setting up model')
        model = get_model(
            settings_model=settings['dnn_training_settings']['model'],
            settings_io=settings['dirs_and_files'],
            output_classes=len(indices_list),
            device=device)
        model.to(device)
        logger_inner.info('Done\n')

        logger_inner.info(f'Model:\n{model}\n')
        logger_inner.info('Total amount of parameters: '
                          f'{sum([i.numel() for i in model.parameters()])}')
        logger_inner.info('Starting training')
        _do_training(
            model=model,
            settings_training=settings['dnn_training_settings']['training'],
            settings_data=settings['dnn_training_settings']['data'],
            settings_io=settings['dirs_and_files'],
            model_file_name=model_file_name,
            model_dir=model_dir,
            indices_list=indices_list)
        logger_inner.info('Training done')

    if settings['workflow']['dnn_evaluation']:
        logger_main.info('Doing evaluation')
        if model is None:
            if not settings['dnn_training_settings']['model']['use_pre_trained_model']:
                raise AttributeError('Mode is set to only evaluation, but'
                                     'is specified not to use a pre-trained model.')

            logger_inner.info('Setting up model')
            model = get_model(
                settings_model=settings['dnn_training_settings']['model'],
                settings_io=settings['dirs_and_files'],
                output_classes=len(indices_list),
                device=device)
            model.to(device)
            logger_inner.info('Model ready')

        logger_inner.info('Starting evaluation')
        _do_evaluation(
            model=model,
            settings_data=settings['dnn_training_settings']['data'],
            settings_io=settings['dirs_and_files'],
            indices_list=indices_list)
        logger_inner.info('Evaluation done')

    if settings['workflow']['dnn_testing']:
        logger_main.info('Doing testing')
        if model is None:
            if not settings['dnn_training_settings']['model']['use_pre_trained_model']:
                raise AttributeError('Mode is not set to train, but'
                                     'is specified not to use a pre-trained model.')

            logger_inner.info('Setting up model')
            model = get_model(
                settings_model=settings['dnn_training_settings']['model'],
                settings_io=settings['dirs_and_files'],
                output_classes=len(indices_list),
                device=device)
            model.to(device)
            logger_inner.info('Model ready')

        logger_inner.info('Starting testing')
        _do_testing(
            model=model,
            settings_data=settings['dnn_training_settings']['data'],
            settings_io=settings['dirs_and_files'],
            indices_list=indices_list)
        logger_inner.info('Testing done')


def main():
    args = get_argument_parser().parse_args()

    file_dir = args.file_dir
    config_file = args.config_file
    file_ext = args.file_ext
    verbose = args.verbose

    settings = file_io.load_yaml_file(Path(
        file_dir, f'{config_file}.{file_ext}'))

    printing.init_loggers(
        verbose=verbose,
        settings=settings['dirs_and_files'])

    logger_main = logger.bind(is_caption=False, indent=0)

    logger_main.info('Starting method only')
    method(settings)
    logger_main.info('Method\'s done')


if __name__ == '__main__':
    main()

# EOF
